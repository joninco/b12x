"""Four-rank graph replay of the opaque DCP channels at capacities up to 64 rows.

Set B12X_RUN_PCIE_DCP_CHANNEL_TEST=1 on a host with four idle GPUs. Every rank
constructs one PCIeDCPAttention and one PCIeDCPTopKPull channel per (capacity,
live rows) pair, runs the eager warmup inside the channels' capture scope as
the vLLM graph manager does, captures query gather, masked LSE combine and
candidate merge into one CUDA graph, and replays it against CPU oracles with
refreshed inputs. Kernel resolution is frozen during capture and the replays,
so a capacity above 16 rows must reuse the geometry compiled at construction.
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    os.getenv("B12X_RUN_PCIE_DCP_CHANNEL_TEST") != "1",
    reason="set B12X_RUN_PCIE_DCP_CHANNEL_TEST=1 to run the four-rank channel test",
)

WORLD_SIZE = 4
TOPK = 2048
# Capacities of captured decode graphs above and below the former 16-row bound,
# each replayed at one row, at half capacity and at full capacity.
CAPACITIES = (16, 24, 32, 64)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _live_rows(capacity: int) -> tuple[int, ...]:
    return tuple(sorted({1, capacity // 2, capacity}))


def _check_channel_pair(group, device, rank, capacity, rows):
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )
    from b12x.comm.pcie.pcie_dcp_attention import PCIeDCPAttention
    from b12x.comm.pcie.pcie_dcp_topk_pull import PAYLOAD_OFFSET, PCIeDCPTopKPull
    from benchmarks.dcp_transport.fixtures import rank_inputs, references

    attention = PCIeDCPAttention(
        process_group=group,
        device=device,
        channel_id=f"channel-test:{capacity}:{rows}",
        max_rows=capacity,
    )
    candidates = PCIeDCPTopKPull(
        process_group=group, device=device, max_rows=capacity, topk=TOPK
    )
    try:
        assert attention.max_rows == capacity
        assert candidates.max_rows == capacity
        assert candidates.slab_bytes == PAYLOAD_OFFSET + capacity * TOPK * 8
        assert attention.allocated_bytes > 0
        query = torch.empty((rows, 8, 576), dtype=torch.bfloat16, device=device)
        gathered = torch.empty((rows, 32, 576), dtype=torch.bfloat16, device=device)
        partial = torch.empty((rows, 32, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((rows, 32), dtype=torch.float32, device=device)
        lengths = torch.empty(rows, dtype=torch.int32, device=device)
        masked_lse = torch.empty_like(lse)
        combined = torch.empty((rows, 8, 512), dtype=torch.bfloat16, device=device)
        packed = torch.empty((rows, TOPK, 2), dtype=torch.float32, device=device)
        selected = torch.empty((rows, TOPK), dtype=torch.int32, device=device)

        def refresh(seed):
            values = [
                rank_inputs(r, rows, seed=seed, topk=TOPK, world_size=WORLD_SIZE)
                for r in range(WORLD_SIZE)
            ]
            local = values[rank]
            local_lengths = torch.isfinite(local.lse).any(dim=1).to(torch.int32)
            # The mask must come from the per-query lengths, not from the
            # -inf values the oracle uses for empty shards.
            query.copy_(local.query)
            partial.copy_(local.partial)
            lse.copy_(local.lse.masked_fill(local_lengths[:, None] == 0, 7.0))
            lengths.copy_(local_lengths)
            packed.copy_(local.packed)
            return references(values, rank)

        def run():
            attention.query(query, gathered)
            attention.combine_masked(partial, lse, lengths, masked_lse, combined)
            candidates.merge(packed, selected)

        expected = refresh(1931)
        graph = torch.cuda.CUDAGraph()
        with attention.capture(), candidates.capture():
            # The eager warmup may resolve the eager launch variants once, as
            # the serving warmup does; capture and replay must not compile.
            run()
            torch.cuda.synchronize(device)
            dist.barrier(group)
            freeze_kernel_resolution(
                f"DCP channels at capacity {capacity} reuse the geometry "
                "compiled at construction"
            )
            try:
                with torch.cuda.graph(graph):
                    run()
            finally:
                unfreeze_kernel_resolution()
        dist.barrier(group)
        freeze_kernel_resolution("DCP channel replays resolve no kernels")
        try:
            pointers = (gathered.data_ptr(), combined.data_ptr(), selected.data_ptr())
            for seed, expected in ((1931, expected), (3911, None)):
                if expected is None:
                    expected = refresh(seed)
                    torch.cuda.synchronize(device)
                    dist.barrier(group)
                graph.replay()
                torch.cuda.synchronize(device)
                first = (gathered.clone(), combined.clone(), selected.clone())
                torch.testing.assert_close(gathered.cpu(), expected[0], rtol=0, atol=0)
                torch.testing.assert_close(
                    combined.cpu(), expected[1], rtol=0.02, atol=0.02
                )
                torch.testing.assert_close(
                    selected.cpu().sort(dim=-1).values, expected[2], rtol=0, atol=0
                )
                for _ in range(5):
                    dist.barrier(group)
                    allocated = torch.cuda.memory_allocated(device)
                    graph.replay()
                    torch.cuda.synchronize(device)
                    assert torch.cuda.memory_allocated(device) == allocated
                    for current, reference in zip(
                        (gathered, combined, selected), first, strict=True
                    ):
                        assert torch.equal(current, reference)
                assert (
                    gathered.data_ptr(),
                    combined.data_ptr(),
                    selected.data_ptr(),
                ) == pointers
                dist.barrier(group)
        finally:
            unfreeze_kernel_resolution()
        del graph
        torch.cuda.synchronize(device)
        dist.barrier(group)
    finally:
        candidates.close()
        attention.close()


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    try:
        group = dist.group.WORLD
        for capacity in CAPACITIES:
            for rows in _live_rows(capacity):
                if rank == 0:
                    print(f"DCP channels: capacity={capacity} rows={rows}", flush=True)
                _check_channel_pair(group, device, rank, capacity, rows)
        if rank == 0:
            print("DCP channels: complete", flush=True)
    finally:
        dist.destroy_process_group()


def test_dcp_channels_replay_exactly_up_to_sixty_four_rows():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"need {WORLD_SIZE} CUDA devices")
    mp.spawn(_worker, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)
