"""CuTe peer candidate selection with deterministic per-thread compaction.

The top-k set uses descending scores with ascending token-id ties. Physical
output order is deterministic for fixed candidate positions: radix pass,
thread index, then each thread's candidate index. It is not score-sorted.
Inputs contain unique nonnegative token ids (or -1 for absent candidates),
exactly representable in float32, and no NaN scores.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from ._dcp_cute_common import block_pair_barrier
from .dcp_candidate_topk import _RankMajorTopKKernel, _block_scan_inclusive_i32


class _PeerTopK(_RankMajorTopKKernel):
    def __init__(self, topk, world_size):
        super().__init__(topk, world_size)
        self.world_size = world_size

    @cute.jit
    def __call__(
        self,
        scores: tuple,
        out: cute.Pointer,
        rows: Int32,
        row_stride: Int64,
        out_stride: Int64,
        stream: CUstream,
    ):
        self.select(scores, out, row_stride, out_stride).launch(
            grid=(rows, 1, 1), block=(self.tb_size, 1, 1), stream=stream
        )

    @cute.kernel
    def select(
        self,
        scores: tuple,
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
            # IEEE signed zeros compare equal and must use the token-id tie break.
            if score == Float32(0):
                score = Float32(0)
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

    @cute.jit
    def _commit_keys(
        self,
        keys,
        output,
        storage,
        tid,
        prefix,
        prefix_bits,
        shift,
        bin_mask,
        threshold,
        should_include_threshold,
    ):
        selected = cute.make_rmem_tensor((self.keys_per_thread,), Int32)
        count = Int32(0)
        for i in cutlass.range_constexpr(self.keys_per_thread):
            key = keys[i]
            bin_idx = Int32((key >> Uint64(shift)) & bin_mask)
            take = self._prefix_matches(key, prefix, prefix_bits) and (
                bin_idx > threshold
                or (should_include_threshold and bin_idx == threshold)
            )
            selected[i] = Int32(take)
            count += selected[i]
        prefix_count = _block_scan_inclusive_i32(
            count,
            cute.arch.lane_idx(),
            cute.arch.warp_idx(),
            storage.warp_totals.get_tensor(cute.make_layout((1, self.warps_per_block))),
            self.warps_per_block,
        )
        committed = storage.committed_count.data_ptr().load()
        destination = committed + prefix_count - count
        for i in cutlass.range_constexpr(self.keys_per_thread):
            if selected[i] != Int32(0):
                if destination < Int32(self.topk):
                    output[destination] = (~cutlass.Uint32(keys[i])).bitcast(Int32)
                destination += Int32(1)
        # All threads read the previous count before one publishes the total.
        cute.arch.sync_threads()
        if tid == Int32(self.tb_size - 1):
            storage.committed_count.data_ptr().store(committed + prefix_count)


def _pointer(dtype, address):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=4)


@cache
def precompile_peer_topk(topk: int, world_size: int, device_index: int):
    """Compile static candidate geometry before freezing kernel resolution."""
    if topk not in (512, 1024, 2048) or world_size not in (2, 4, 8):
        raise ValueError("Peer top-k requires K=512/1024/2048 and ranks=2/4/8")
    key = (topk, world_size, device_index)
    raise_if_kernel_resolution_frozen("cute.compile", target=_PeerTopK, cache_key=key)
    with torch.cuda.device(device_index):
        return b12x_compile(
            _PeerTopK(topk, world_size),
            tuple(_pointer(Float32, 16) for _ in range(world_size)),
            _pointer(Int32, 16),
            1,
            1,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts("comm.pcie.peer_topk", 3, *key),
        )


def select_peer_topk(
    pointers: tuple[int, ...], out: torch.Tensor, row_stride: int
) -> None:
    """Consume channel-owned published slabs into caller-owned output storage.

    The caller must fence publication and prevent overwrite until the consumer
    finishes. Every pointer addresses float32 [capacity,K,2] score/id storage.
    """
    if (
        out.ndim != 2
        or out.dtype != torch.int32
        or not out.is_cuda
        or out.stride(1) != 1
        or out.stride(0) < out.shape[1]
        or row_stride < out.shape[1] * 2
        or any(p <= 0 or p % 4 for p in pointers)
    ):
        raise ValueError("Peer candidate pointers or output storage are invalid")
    rows, topk = out.shape
    fn = precompile_peer_topk(topk, len(pointers), out.device.index)
    with torch.cuda.device(out.device):
        fn(
            tuple(_pointer(Float32, p) for p in pointers),
            _pointer(Int32, out.data_ptr()),
            rows,
            row_stride,
            out.stride(0),
            current_cuda_stream(),
        )


BLOCKS = 16


class _CandidatePublish:
    def __init__(self, rank, topk):
        self.rank, self.words = rank, topk * 2

    @cute.jit
    def __call__(
        self,
        signals: tuple,
        source0: cute.Pointer,
        destination0: cute.Pointer,
        rows: Int32,
        stream: CUstream,
    ):
        self.publish(signals, source0, destination0, rows).launch(
            grid=(BLOCKS, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def publish(
        self,
        signals: tuple,
        source0: cute.Pointer,
        destination0: cute.Pointer,
        rows: Int32,
    ):
        block_pair_barrier(
            signals,
            self_signal=signals[self.rank],
            rank=self.rank,
            world_size=4,
            max_blocks=BLOCKS,
        )
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        start = Int64(block) * Int64(256) + Int64(thread)
        for offset in range(start, Int64(rows) * Int64(self.words), BLOCKS * 256):
            (destination0 + offset).store((source0 + offset).load())
        block_pair_barrier(
            signals,
            self_signal=signals[self.rank],
            rank=self.rank,
            world_size=4,
            max_blocks=BLOCKS,
        )


@cache
def precompile_candidate_publication(rank: int, topk: int, device_index: int):
    """Resolve the fixed 16-CTA publication geometry before graph capture."""
    if rank not in range(4) or topk not in (512, 1024, 2048):
        raise ValueError(
            "Candidate publication requires four ranks and K=512/1024/2048"
        )
    key = (rank, topk, device_index)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=_CandidatePublish, cache_key=key
    )
    with torch.cuda.device(device_index):
        return b12x_compile(
            _CandidatePublish(rank, topk),
            tuple(_pointer(Uint32, 16) for _ in range(4)),
            *(_pointer(Uint32, 16) for _ in range(2)),
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "comm.pcie.candidate_publish", 2, *key
            ),
        )
