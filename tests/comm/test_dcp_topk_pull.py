"""Peer candidate selection preserves exact sets and deterministic graph output."""

import pytest
import torch

from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)
from b12x.comm.pcie._dcp_topk_pull_cute import (
    precompile_peer_topk,
    select_peer_topk,
)
from benchmarks.dcp_transport.fixtures import rank_inputs, references


@pytest.mark.parametrize("mismatch", ["topk", "capacity"])
def test_compiled_candidate_merge_enforces_planned_geometry(monkeypatch, mismatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.comm.pcie import pcie_dcp_topk_pull as module

    device = torch.device("cuda", torch.cuda.current_device())
    channel = object.__new__(module.PCIeDCPTopKPull)
    channel.device, channel.rank = device, 0
    channel.topk, channel.max_rows = 2048, 4
    channel._state = torch.empty(
        module.PAYLOAD_OFFSET + 4 * 2048 * 8, dtype=torch.uint8, device=device
    )
    channel._peer_slabs = [channel._state.data_ptr()] * 4
    rows, topk = (1, 512) if mismatch == "topk" else (5, 2048)
    packed = torch.empty((rows, topk, 2), device=device)
    out = torch.empty((rows, topk), dtype=torch.int32, device=device)

    def unexpected_launch(*args, **kwargs):
        raise AssertionError("Invalid geometry reached publication kernel resolution")

    monkeypatch.setattr(module, "precompile_candidate_publication", unexpected_launch)
    compiled = torch.compile(channel.merge, backend="eager", fullgraph=True)
    with pytest.raises(ValueError, match="geometry are incompatible"):
        compiled(packed, out)


@pytest.mark.parametrize("rows", [17, 32, 64])
def test_compiled_candidate_merge_accepts_capacities_above_sixteen_rows(
    monkeypatch, rows
):
    """A 64-row channel publishes and selects live rows up to its capacity."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.comm.pcie import pcie_dcp_topk_pull as module

    device = torch.device("cuda", torch.cuda.current_device())
    channel = object.__new__(module.PCIeDCPTopKPull)
    channel.device, channel.rank = device, 0
    channel.topk, channel.max_rows = 2048, 64
    channel._state = torch.empty(
        module.PAYLOAD_OFFSET + 64 * 2048 * 8, dtype=torch.uint8, device=device
    )
    channel._peer_slabs = [channel._state.data_ptr()] * 4
    packed = torch.empty((rows, 2048, 2), device=device)
    out = torch.empty((rows, 2048), dtype=torch.int32, device=device)
    launches = []

    def publication(rank, topk, device_index):
        return lambda signals, source, destination, live_rows, stream: launches.append(
            ("publish", live_rows)
        )

    def selection(pointers, output, row_stride):
        launches.append(("select", output.shape[0]))

    monkeypatch.setattr(module, "precompile_candidate_publication", publication)
    monkeypatch.setattr(module, "select_peer_topk", selection)
    compiled = torch.compile(channel.merge, backend="eager", fullgraph=True)
    compiled(packed, out)
    assert launches == [("publish", rows), ("select", rows)]
    with pytest.raises(ValueError, match="geometry are incompatible"):
        compiled(
            torch.empty((65, 2048, 2), device=device),
            torch.empty((65, 2048), dtype=torch.int32, device=device),
        )


def test_peer_selector_ties_signed_zero_by_token_id():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    topk = 512
    ids = torch.arange(topk * 4).reshape(4, 1, topk).float()
    scores = torch.zeros_like(ids)
    scores[0] = -0.0
    slabs = torch.stack((scores, ids), -1).to(device)
    out = torch.empty((1, topk), dtype=torch.int32, device=device)
    precompile_peer_topk(topk, 4, device.index)
    select_peer_topk(tuple(slabs[r].data_ptr() for r in range(4)), out, topk * 2)
    torch.testing.assert_close(
        out.cpu().sort().values, torch.arange(topk, dtype=torch.int32)[None]
    )


@pytest.mark.parametrize("topk", [512, 2048])
def test_peer_selector_exact_repeatable_all_live_rows(topk):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    compiled = precompile_peer_topk(topk, 4, device.index)
    capacity = 64
    slabs = torch.empty((4, capacity, topk + 8, 2), device=device)
    pointers = tuple(slabs[r].data_ptr() for r in range(4))
    output = torch.full((capacity, topk + 8), -7, dtype=torch.int32, device=device)
    freeze_kernel_resolution(
        "Peer top-k reuses static geometry across rows 1 through 64"
    )
    try:
        # Rows 1 through 16 plus the larger uniform decode graph sizes up to
        # the 64-row transport capacity.
        for rows in (*range(1, 17), 24, 32, 40, 48, 56, 64):
            assert precompile_peer_topk(topk, 4, device.index) is compiled
            out = output[:rows, :topk]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                select_peer_topk(pointers, out, slabs.stride(1))
            address = out.data_ptr()
            for seed in (951, 1953):
                ranks = [rank_inputs(r, rows, topk=topk, seed=seed) for r in range(4)]
                expected = references(ranks, 0)[2].sort(dim=1).values
                for rank, values in enumerate(ranks):
                    slabs[rank, :rows, :topk].copy_(values.packed)
                graph.replay()
                torch.cuda.synchronize()
                baseline = out.clone()
                torch.testing.assert_close(out.cpu().sort(dim=1).values, expected)
                allocated = torch.cuda.memory_allocated()
                for _ in range(30):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, baseline)
                    assert out.data_ptr() == address
                    assert torch.cuda.memory_allocated() == allocated
                assert torch.all(output[:, topk:] == -7)
    finally:
        unfreeze_kernel_resolution()
