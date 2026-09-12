"""Exact candidate consumers for the standalone transport comparison.

Status: research-only. Owner staging exposes separate score/id planes. Peer
publication exposes one interleaved score/id slab per rank. Both consumers use
the package radix selector's score-descending, token-id-ascending set contract;
neither materializes an intermediate candidate layout. Callers synchronize
publication and keep all input storage alive until the consumer completes.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, Uint64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.comm.pcie.dcp_candidate_topk import _RankMajorTopKKernel


class _TransportTopK(_RankMajorTopKKernel):
    def __init__(self, topk, world_size, layout):
        super().__init__(topk, world_size)
        self.world_size = world_size
        self.layout = layout

    @cute.jit
    def __call__(
        self,
        scores: tuple,
        ids: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        row_stride: Int64,
        out_stride: Int64,
        stream: CUstream,
    ):
        self.select(scores, ids, out, row_stride, out_stride).launch(
            grid=(rows, 1, 1), block=(self.tb_size, 1, 1), stream=stream
        )

    @cute.kernel
    def select(
        self,
        scores: tuple,
        ids: cute.Pointer,
        out: cute.Pointer,
        row_stride: Int64,
        out_stride: Int64,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = Int64(row)
        output = cute.make_tensor(
            out + row * out_stride, cute.make_layout((self.topk,))
        )
        keys = cute.make_rmem_tensor((self.keys_per_thread,), Uint64)
        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage, 8)
        for col in range(tid, self.topk, self.tb_size):
            output[col] = Int32(-1)
        for i in cutlass.range_constexpr(self.keys_per_thread):
            col = Int64(tid + i * self.tb_size)
            if cutlass.const_expr(self.layout == "owner_planes"):
                offset = row * row_stride + col
                score = (scores[0] + offset).load()
                token_id = (ids + offset).load()
            else:
                source = col // Int64(self.topk)
                address = Int64(scores[0].toint())
                for peer in cutlass.range_constexpr(1, self.world_size):
                    if source == Int64(peer):
                        address = Int64(scores[peer].toint())
                pointer = cute.make_ptr(
                    Float32, address, cute.AddressSpace.gmem, assumed_align=4
                )
                offset = row * row_stride + (col % Int64(self.topk)) * Int64(2)
                score = (pointer + offset).load()
                token_id = Int32((pointer + offset + Int64(1)).load())
            keys[i] = self._stable_key(score, token_id)
        if tid == Int32(0):
            storage.committed_count.data_ptr().store(Int32(0))
            storage.prefix_s.data_ptr().store(Uint64(0))
        cute.arch.sync_threads()
        step = Int32(0)
        finished = Int32(0)
        while finished == Int32(0) and step < Int32(self.radix_passes - 1):
            finished = self._radix_pass(
                keys, output, storage, tid, step, self.radix_bits, False
            )
            step += Int32(1)
        if finished == Int32(0):
            self._radix_pass(
                keys,
                output,
                storage,
                tid,
                Int32(self.radix_passes - 1),
                self.final_radix_bits,
                True,
            )


def _pointer(dtype, address):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=4)


@cache
def precompile(topk: int, world_size: int, layout: str, device_index: int):
    """Resolve a consumer using only static geometry and storage layout."""
    if topk not in (512, 1024, 2048) or world_size not in (2, 4, 8):
        raise ValueError("Candidate consumers require K=512/1024/2048, ranks=2/4/8")
    if layout not in ("owner_planes", "peer_pairs"):
        raise ValueError(f"Unsupported candidate layout: {layout}")
    key = (topk, world_size, layout, device_index)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=_TransportTopK, cache_key=key
    )
    with torch.cuda.device(device_index):
        return b12x_compile(
            _TransportTopK(topk, world_size, layout),
            tuple(_pointer(Float32, 16) for _ in range(world_size)),
            _pointer(Int32, 16),
            _pointer(Int32, 16),
            1,
            1,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "benchmark.dcp_transport.topk", 1, *key
            ),
        )


def select_owner(
    indices: torch.Tensor, scores: torch.Tensor, out: torch.Tensor, world_size: int
) -> None:
    """Select from owner-local score/id planes without repacking."""
    rows, topk = out.shape
    if (
        indices.shape != scores.shape
        or indices.shape != (rows, topk * world_size)
        or indices.dtype != torch.int32
        or scores.dtype != torch.float32
        or out.dtype != torch.int32
        or not out.is_cuda
        or indices.device != out.device
        or scores.device != out.device
        or indices.stride() != scores.stride()
        or indices.stride(1) != 1
        or out.stride(1) != 1
    ):
        raise ValueError("Owner candidate planes have incompatible shape or storage")
    fn = precompile(topk, world_size, "owner_planes", out.device.index)
    fn(
        tuple(_pointer(Float32, scores.data_ptr()) for _ in range(world_size)),
        _pointer(Int32, indices.data_ptr()),
        _pointer(Int32, out.data_ptr()),
        rows,
        scores.stride(0),
        out.stride(0),
        current_cuda_stream(),
    )


def select_peers(pointers: tuple[int, ...], out: torch.Tensor, row_stride: int) -> None:
    """Consume published peer (score,id) slabs with ordinary global loads.

    The channel owns the pointers and publication/overwrite barriers. Each slab
    must cover every output row at the supplied float32-element row stride.
    """
    rows, topk = out.shape
    if (
        out.dtype != torch.int32
        or not out.is_cuda
        or out.stride(1) != 1
        or row_stride < topk * 2
        or any(p <= 0 or p % 4 for p in pointers)
    ):
        raise ValueError("Peer candidate pointers or output storage are invalid")
    fn = precompile(topk, len(pointers), "peer_pairs", out.device.index)
    fn(
        tuple(_pointer(Float32, p) for p in pointers),
        _pointer(Int32, out.data_ptr()),
        _pointer(Int32, out.data_ptr()),
        rows,
        row_stride,
        out.stride(0),
        current_cuda_stream(),
    )
