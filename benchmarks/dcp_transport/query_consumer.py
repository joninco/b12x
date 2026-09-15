"""Sparse attention consumer for published DCP queries.

Status: research-only. The consumer uses UnifiedDecodeKernel's GLM FP8 math,
32 query heads, and split-K partial outputs. Its local and peer variants differ
only in query memory access. The caller owns publication/overwrite synchronization
and keeps every peer allocation alive until the attention kernel completes.
No serving dispatch selects this benchmark entry.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint8

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.attention._shared.mla.kernel import UnifiedDecodeKernel
from b12x.attention._shared.mla.smem import make_smem_layout
from b12x.attention._shared.mla.traits import (
    ComputeMode,
    ModelType,
    ScaleFormat,
    make_unified_traits,
)


class _QueryConsumer(UnifiedDecodeKernel):
    def __init__(self, *, topk, splits, capacity, peer_reads):
        traits = make_unified_traits(
            ModelType.GLM_NSA, ComputeMode.FP8, ScaleFormat.ARBITRARY_FP32
        )
        super().__init__(
            traits,
            make_smem_layout(traits),
            page_block_size=64,
            chunks_per_split=(topk + 64 * splits - 1) // (64 * splits),
            h_blocks=2,
            num_splits=splits,
            num_heads=32,
            q_head_dim=576,
            topk=topk,
            extra_topk=0,
            q_stride=(32 * 576, 576, 1),
            swa_indices_stride0=topk,
            extra_indices_stride0=topk,
            mid_out_stride=(32 * splits * 512, splits * 512, 512, 1),
            mid_lse_stride=(32 * splits, splits, 1),
            valid_hpb=16,
            vector_q=True,
        )
        self.capacity = capacity
        self.peer_reads = peer_reads

    @cute.jit
    def __call__(
        self,
        query: cute.Pointer,
        peers: tuple,
        kv: cute.Pointer,
        indices: cute.Pointer,
        out: cute.Pointer,
        lse: cute.Pointer,
        rows: Int32,
        active_candidates: Int32,
        stream: CUstream,
    ):
        self.consume(query, peers, kv, indices, out, lse, active_candidates).launch(
            grid=(rows, self.h_blocks, self.num_splits),
            block=(self.block_threads, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def consume(
        self,
        query: cute.Pointer,
        peers: tuple,
        kv: cute.Pointer,
        indices: cute.Pointer,
        out: cute.Pointer,
        lse: cute.Pointer,
        active_candidates: Int32,
    ):
        q = cute.make_tensor(
            query, cute.make_layout((self.capacity, 32, 576), stride=(32 * 576, 576, 1))
        )
        cache = cute.make_tensor(kv, cute.make_layout((self.topk * 656,)))
        selection = cute.make_tensor(
            indices, cute.make_layout((self.capacity, self.topk), stride=(self.topk, 1))
        )
        output = cute.make_tensor(
            out,
            cute.make_layout(
                (self.capacity, 32, self.num_splits, 512),
                stride=(32 * self.num_splits * 512, self.num_splits * 512, 512, 1),
            ),
        )
        logsumexp = cute.make_tensor(
            lse,
            cute.make_layout(
                (self.capacity, 32, self.num_splits),
                stride=(32 * self.num_splits, self.num_splits, 1),
            ),
        )
        peer_query = ()
        if cutlass.const_expr(self.peer_reads):
            peer_query = peers
        self._kernel_body(
            q,
            cache,
            selection,
            output,
            logsumexp,
            Float32(576**-0.5 * 1.4426950408889634),
            Float32(1),
            active_candidates,
            Int64(64 * 656),
            cache,
            selection,
            Int32(0),
            Int32(0),
            Int64(64 * 656),
            selection,
            selection,
            has_extra=False,
            per_token_len=False,
            peer_query_pointers=peer_query,
        )


def _pointer(dtype, address):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=16)


def precompile(*, topk=2048, splits=32, capacity=16, peer_reads=True, device_index=0):
    """Compile one planned capacity and split geometry before capture."""
    if topk not in (512, 2048) or not 1 <= splits <= 32 or capacity != 16:
        raise ValueError(
            "Query consumer requires K=512/2048, 1..32 splits, capacity=16"
        )
    return _compile(topk, splits, capacity, peer_reads, device_index)


@cache
def _compile(topk, splits, capacity, peer_reads, device_index):
    key = (topk, splits, capacity, peer_reads, device_index)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=_QueryConsumer, cache_key=key
    )
    with torch.cuda.device(device_index):
        return b12x_compile(
            _QueryConsumer(
                topk=topk, splits=splits, capacity=capacity, peer_reads=peer_reads
            ),
            _pointer(BFloat16, 16),
            tuple(_pointer(BFloat16, 16) for _ in range(4)),
            _pointer(Uint8, 16),
            _pointer(Int32, 16),
            _pointer(BFloat16, 16),
            _pointer(Float32, 16),
            1,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "benchmark.dcp_transport.query_consumer", 2, *key
            ),
        )


def consume(query, peers, kv, indices, out, lse, *, active_candidates=None):
    """Write split-K partials using local queries or four published peer slabs."""
    rows, heads, dim = query.shape
    topk = indices.shape[1]
    splits = out.shape[2]
    if (
        heads != 32
        or dim != 576
        or not 1 <= rows <= 16
        or query.dtype != torch.bfloat16
        or not query.is_cuda
        or indices.shape != (rows, topk)
        or indices.dtype != torch.int32
        or out.shape != (rows, 32, splits, 512)
        or out.dtype != torch.bfloat16
        or lse.shape != (rows, 32, splits)
        or lse.dtype != torch.float32
        or kv.dtype != torch.uint8
        or kv.numel() < topk * 656
        or any(
            t.device != query.device or not t.is_contiguous() or t.data_ptr() % 16
            for t in (query, kv, indices, out, lse)
        )
        or (peers and (len(peers) != 4 or any(p <= 0 or p % 16 for p in peers)))
    ):
        raise ValueError("Query consumer geometry, dtype, device or alignment mismatch")
    active = topk if active_candidates is None else active_candidates
    if not 0 <= active <= topk:
        raise ValueError("Active candidate count must fit the planned top-k")
    fn = precompile(
        topk=topk,
        splits=splits,
        peer_reads=bool(peers),
        device_index=query.device.index,
    )
    pointers = peers or (query.data_ptr(),) * 4
    fn(
        _pointer(BFloat16, query.data_ptr()),
        tuple(_pointer(BFloat16, p) for p in pointers),
        _pointer(Uint8, kv.data_ptr()),
        _pointer(Int32, indices.data_ptr()),
        _pointer(BFloat16, out.data_ptr()),
        _pointer(Float32, lse.data_ptr()),
        rows,
        active,
        current_cuda_stream(),
    )
