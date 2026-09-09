"""Complete exchange paths for the eight-rank DCP graph benchmark."""

from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Callable

import torch
import triton
import triton.language as tl

from b12x.comm.pcie.dcp_candidate_topk import (
    pack_dcp_candidates,
    precompile_rank_major_topk,
    rank_major_topk,
)
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2APool
from b12x.comm.pcie.pcie_dcp_topk import PCIeDCPTopKOwnerExchange
from benchmarks.dcp_transport.fixtures import rank_inputs, references
from benchmarks.dcp_transport.selectors import precompile, select_owner


@dataclass
class Case:
    name: str
    operation: str
    run: Callable
    outputs: tuple[torch.Tensor, ...]
    expected: tuple[torch.Tensor, ...]
    capture: Callable = nullcontext
    resources: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def check(self):
        for actual, expected in zip(self.outputs, self.expected, strict=True):
            actual = actual.cpu()
            if self.operation == "candidates":
                actual = actual.sort(dim=-1).values
            tolerance = 0 if expected.dtype in (torch.int32, torch.int64) else 0.02
            if self.operation == "query":
                tolerance = 0
            torch.testing.assert_close(
                actual, expected, rtol=tolerance, atol=tolerance, equal_nan=False
            )

    def close(self):
        for resource in reversed(self.resources):
            close = getattr(resource, "close", None)
            if close:
                close()


def _pool(group, device, *, heads=32, dim=512, query_dim=576):
    pool = PCIeDCPA2APool.from_process_group(
        process_group=group,
        device=device,
        max_batch_size=16,
        total_heads=heads,
        head_dim=dim,
        query_head_dim=query_dim,
    )
    pool.prepare_channels(("benchmark",))
    pool.prepare_graph_all_gather_heads(channel_id="benchmark")
    pool.prepare_graph_lse_reduce_scatter(dtype=torch.bfloat16, channel_id="benchmark")
    return pool


def attention_cases(kind, group, gpu_group, device, rows, rank):
    ranks = [rank_inputs(r, rows) for r in range(4)]
    expected_query, expected_output, _ = references(ranks, rank)
    local = ranks[rank]
    query = local.query.to(device)
    partial, lse = local.partial.to(device), local.lse.to(device)
    output = torch.empty((rows, 8, 512), dtype=torch.bfloat16, device=device)
    details = {}
    query_name, pair_name = kind + "_query", kind + "_pair"
    if kind == "b12x":
        pool = _pool(group, device)
        gathered = torch.empty((rows, 32, 576), dtype=torch.bfloat16, device=device)

        def gather():
            pool.all_gather_heads(query, gathered, channel_id="benchmark")

        def combine():
            pool.lse_reduce_scatter(partial, lse, output, channel_id="benchmark")

        def capture():
            return pool.capture(channel_id="benchmark")

        resources = [pool]
    elif kind == "native":
        from vllm.v1.attention.ops.dcp import (
            DirectDCPA2AWorkspace,
            DirectDCPQGatherWorkspace,
        )

        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        multicast = _SymmetricMemory.has_multicast_support(
            DeviceType.CUDA, device.index
        )
        if multicast:
            q_workspace = DirectDCPQGatherWorkspace(gpu_group, device, 16, 8, 576)
            gathered = q_workspace.final_query[0, :rows]
            q_signal = q_workspace.received_signal[0]
            q_completion = q_workspace.completion[0]
            q_epoch = q_workspace.epoch[:1]
            q_multicast, signal_multicast = q_workspace.multicast_ptrs[0]

            def gather():
                torch.ops._C.direct_dcp_q_gather(
                    query,
                    gathered,
                    q_signal,
                    q_completion,
                    q_epoch,
                    4,
                    rank,
                    16,
                    32,
                    q_multicast,
                    signal_multicast,
                )
        else:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

            q_workspace = PyNcclCommunicator(group, device)
            if q_workspace.disabled:
                raise RuntimeError("The query all-gather baseline requires PyNccl")
            rank_major = torch.empty(
                (4, rows, 8, 576), dtype=query.dtype, device=device
            )
            gathered = torch.empty((rows, 32, 576), dtype=query.dtype, device=device)
            destination = gathered.view(rows, 4, 8, 576)
            source = rank_major.movedim(0, 1)

            def gather():
                q_workspace.all_gather(rank_major, query)
                # Same rank-major-to-head-major copy as GroupCoordinator's
                # dimension-1 all-gather, with its destination preallocated.
                destination.copy_(source)

            query_name = "nccl_query"
            pair_name = "nccl_query_native_combine"
            details = {
                "compiled_query_status": "unsupported",
                "compiled_query_reason": "NVLS symmetric-memory multicast unavailable",
                "query_transport": "PyNccl all-gather and rank-to-head layout copy",
            }
        o_workspace = DirectDCPA2AWorkspace(gpu_group, device, 16, 8, 512)
        o_args = (
            o_workspace.peer_output_ptrs[0],
            o_workspace.peer_lse_ptrs[0],
            o_workspace.peer_signal_ptrs[0],
            o_workspace.received_output[0],
            o_workspace.received_lse[0],
            o_workspace.received_signal[0],
            o_workspace.epoch[:1],
        )
        local_lengths = torch.ones(rows, dtype=torch.int32, device=device)
        if rank:
            local_lengths[0] = 0
        if rows > 1:
            local_lengths[1] = 0
        query_starts = torch.arange(rows + 1, dtype=torch.int32, device=device)

        def combine():
            # Invoke the compiled operation with caller-owned output. The
            # serving wrapper allocates the result tensor before dispatch.
            torch.ops._C.direct_dcp_a2a_lse_reduce(
                partial,
                lse,
                local_lengths,
                query_starts,
                *o_args,
                output,
                4,
                rank,
                16,
                True,
            )

        capture = nullcontext
        resources = [q_workspace, o_workspace]
    else:
        raise ValueError(f"Unknown attention transport {kind}")

    def pair():
        gather()
        combine()

    # Resources are closed by the final case after all three serial captures
    # and their replays finish; every closure retains its input tensors.
    return [
        Case(
            query_name,
            "query",
            gather,
            (gathered,),
            (expected_query,),
            capture,
            details=details,
        ),
        Case(
            kind + "_combine",
            "combine",
            combine,
            (output,),
            (expected_output,),
            capture,
        ),
        Case(
            pair_name,
            "pair",
            pair,
            (gathered, output),
            (expected_query, expected_output),
            capture,
            resources,
            details,
        ),
    ]


@triton.jit
def _restore_owner_rows(
    source, output, rows, owner_rows, topk: tl.constexpr, block: tl.constexpr
):
    offset = (tl.program_id(0) * block + tl.arange(0, block)).to(tl.int64)
    row = offset // topk
    owner = row // owner_rows
    owner_row = row % owner_rows
    source_offset = (owner_row * 4 + owner) * topk + offset % topk
    value = tl.load(source + source_offset, mask=row < rows, other=-1)
    tl.store(output + offset, value, mask=row < rows)


def candidate_case(kind, group, device, rows, rank):
    padded_rows = max(4, rows) if kind == "owner" else rows
    ranks = [rank_inputs(r, padded_rows) for r in range(4)]
    expected = references(ranks, rank)[2][:rows]
    local = ranks[rank]
    ids, scores = local.local_ids.to(device), local.scores.to(device)
    packed = torch.empty((padded_rows, 2048, 2), device=device)
    output = torch.empty((padded_rows, 2048), dtype=torch.int32, device=device)
    if kind == "rank_major":
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        communicator = PyNcclCommunicator(group, device)
        if communicator.disabled:
            raise RuntimeError("The rank-major baseline requires PyNccl")
        gathered = torch.empty((4, rows, 2048, 2), device=device)
        precompile_rank_major_topk(2048, 4, device)

        def run():
            pack_dcp_candidates(ids, scores, packed, rank, 4, 1)
            communicator.all_gather(gathered, packed)
            rank_major_topk(gathered, output)

        return Case(
            "rank_major_candidates",
            "candidates",
            run,
            (output,),
            (expected,),
            resources=[communicator],
            details={"transport": "PyNccl rank-major all-gather"},
        )
    if kind != "owner":
        raise ValueError(f"Unknown candidate transport {kind}")
    owner_rows = padded_rows // 4
    exchange = PCIeDCPTopKOwnerExchange.from_process_group(
        process_group=group,
        device=device,
        max_rows=16,
        topk=2048,
    )
    exchange.prepare_graph()
    precompile(2048, 4, "owner_planes", device.index)
    # One head per rank redistributes selected rows. The live owner-row count
    # is a runtime batch extent; it never becomes a compile-key head count.
    redistribution = _pool(group, device, heads=4, dim=4096, query_dim=4096)
    selected = torch.empty((owner_rows, 2048), dtype=torch.int32, device=device)
    # The gather copies opaque 16-byte vectors. Reinterpret the int32 ids as
    # BF16 storage without conversion so its supported dtype interface retains
    # every id bit, including the -1 sentinel.
    selected_heads = selected.view(torch.bfloat16).view(owner_rows, 1, 4096)
    gathered = torch.empty((owner_rows, 4, 2048), dtype=torch.int32, device=device)
    gathered_heads = gathered.view(torch.bfloat16)
    global_ids = local.packed[..., 1].int().to(device)

    def run():
        # Index conversion is included, just as in the all-gather path. The
        # owner's transport takes separate planes, so no packed buffer is read.
        _global_ids[(padded_rows,)](ids, global_ids, rank, 2048)
        owner_ids, owner_scores = exchange.stage_candidates(global_ids, scores)
        select_owner(owner_ids, owner_scores, selected, 4)
        redistribution.all_gather_heads(
            selected_heads, gathered_heads, channel_id="benchmark"
        )
        _restore_owner_rows[(triton.cdiv(padded_rows * 2048, 512),)](
            gathered,
            output,
            padded_rows,
            owner_rows,
            2048,
            512,
        )

    @contextmanager
    def capture():
        with ExitStack() as stack:
            stack.enter_context(exchange.capture())
            stack.enter_context(redistribution.capture(channel_id="benchmark"))
            yield

    return Case(
        "owner_candidates",
        "candidates",
        run,
        (output[:rows],),
        (expected,),
        capture,
        [exchange, redistribution],
        {
            "padded_rows": padded_rows,
            "candidate_transport": "peer stores in owner staging",
            "result_transport": "plain peer loads and row-order restoration",
        },
    )


@triton.jit
def _global_ids(local, output, rank: tl.constexpr, topk: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, topk)
    ids = tl.load(local + row * topk + col)
    tl.store(output + row * topk + col, tl.where(ids >= 0, ids * 4 + rank, -1))
