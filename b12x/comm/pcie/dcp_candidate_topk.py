# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact top-k selection from rank-major DCP candidates without repacking.

Input is float32 [rank, row, candidate, (score, global token id)]. Token ids
must be exactly representable in float32; negative ids mark absent candidates.
The selected set orders scores descending and breaks ties by smaller token id.
Positive zero precedes negative zero, matching the IEEE bit-key ordering.
Output positions within that set are unspecified. Valid token ids must be unique
within each row across ranks. NaN scores are unsupported.
"""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
import triton
import triton.language as tl
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, Uint32, Uint64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


@cute.jit
def _warp_scan_inclusive_i32(val: Int32, lane: Int32) -> Int32:
    for i in cutlass.range_constexpr(cute.arch.WARP_SIZE.bit_length() - 1):
        offset = 1 << i
        partial = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial
    return val


@cute.jit
def _block_scan_inclusive_i32(
    val: Int32,
    lane: Int32,
    warp_id: Int32,
    warp_scratch: cute.Tensor,
    warps_per_block: int,
) -> Int32:
    prefix = _warp_scan_inclusive_i32(val, lane)
    if lane == Int32(cute.arch.WARP_SIZE - 1):
        warp_scratch[0, warp_id] = prefix
    cute.arch.sync_threads()

    if warp_id == Int32(0):
        warp_total = Int32(0)
        if lane < Int32(warps_per_block):
            warp_total = warp_scratch[0, lane]
        warp_prefix = _warp_scan_inclusive_i32(warp_total, lane)
        if lane < Int32(warps_per_block):
            warp_scratch[0, lane] = warp_prefix - warp_total
    cute.arch.sync_threads()

    return prefix + warp_scratch[0, warp_id]


class _RankMajorTopKKernel:
    tb_size = 512
    hist_bins = 2048
    radix_bits = (hist_bins - 1).bit_length()
    assert hist_bins == 1 << radix_bits
    key_bits = Uint64.width
    radix_passes = (key_bits + radix_bits - 1) // radix_bits
    final_radix_bits = key_bits - radix_bits * (radix_passes - 1)
    hist_chunks = (hist_bins + tb_size - 1) // tb_size
    warps_per_block = tb_size // cute.arch.WARP_SIZE

    def __init__(self, topk: int, world_size: int):
        num_candidates = topk * world_size
        assert num_candidates % self.tb_size == 0, (
            "_RankMajorTopKKernel requires candidate count "
            f"to be a multiple of {self.tb_size}, got {num_candidates}"
        )
        self.topk = topk
        self.keys_per_thread = num_candidates // self.tb_size

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[Int32, self.hist_bins]
            committed_count: cute.struct.MemRange[Int32, 1]
            running_count: cute.struct.MemRange[Int32, 1]
            threshold_bin: cute.struct.MemRange[Int32, 1]
            threshold_found: cute.struct.MemRange[Int32, 1]
            include_threshold_bin: cute.struct.MemRange[Int32, 1]
            prefix_s: cute.struct.Align[cute.struct.MemRange[Uint64, 1], 8]
            warp_totals: cute.struct.MemRange[Int32, self.warps_per_block]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        gathered: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        rank_stride: Int64,
        out_stride: Int64,
        stream: CUstream,
    ):
        grid = (rows, 1, 1)
        self.kernel(gathered, out, rank_stride, out_stride).launch(
            grid=grid,
            block=(self.tb_size, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _stable_key(self, score: Float32, token_id: Int32) -> Uint64:
        bits = score.bitcast(Uint32)
        mask = Uint32(0x80000000)
        if (bits & Uint32(0x80000000)) != Uint32(0):
            mask = Uint32(0xFFFFFFFF)
        score_key = Uint64(bits ^ mask) << Uint64(32)
        id_key = Uint64(~Uint32(token_id))
        key = score_key | id_key
        if token_id < Int32(0):
            key = Uint64(0)
        return key

    @cute.jit
    def _prefix_matches(
        self,
        key: Uint64,
        prefix: Uint64,
        prefix_bits: Int32,
    ):
        matches = prefix_bits == Int32(0)
        if prefix_bits != Int32(0):
            shift = Int32(self.key_bits) - prefix_bits
            matches = (key >> Uint64(shift)) == (prefix >> Uint64(shift))
        return matches

    @cute.jit
    def _radix_pass(
        self,
        keys: cute.Tensor,
        output: cute.Tensor,
        storage,
        tid: Int32,
        step: Int32,
        bits: int,
        is_final_pass: bool,
    ):
        hist_smem = storage.hist.get_tensor(cute.make_layout((self.hist_bins,)))
        committed_count_smem = storage.committed_count.data_ptr()
        running_count_smem = storage.running_count.data_ptr()
        threshold_bin_smem = storage.threshold_bin.data_ptr()
        threshold_found_smem = storage.threshold_found.data_ptr()
        include_threshold_bin_smem = storage.include_threshold_bin.data_ptr()
        prefix_smem = storage.prefix_s.data_ptr()
        warp_totals_smem = storage.warp_totals.get_tensor(
            cute.make_layout((1, self.warps_per_block))
        )

        prefix_bits = step * Int32(self.radix_bits)
        num_bins = 1 << bits
        block_scan_iterations = (num_bins + self.tb_size - 1) // self.tb_size
        shift = Int32(self.key_bits) - prefix_bits - Int32(bits)
        bin_mask = Uint64(num_bins - 1)
        prefix = prefix_smem.load()

        for chunk in cutlass.range_constexpr(self.hist_chunks):
            hist_smem[tid + Int32(chunk * self.tb_size)] = Int32(0)
        if tid == Int32(0):
            running_count_smem.store(committed_count_smem.load())
            include_threshold_bin_smem.store(Int32(0))
            threshold_found_smem.store(Int32(0))
        cute.arch.sync_threads()

        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            key = keys[key_idx]
            if self._prefix_matches(key, prefix, prefix_bits):
                bin_idx = Int32((key >> Uint64(shift)) & bin_mask)
                cute.arch.atomic_add(
                    hist_smem.iterator + bin_idx,
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
        cute.arch.sync_threads()

        lane = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()
        # Each iteration scans one tb_size-wide slice of bins, high to low.
        iter = Int32(0)
        threshold_found = threshold_found_smem.load()
        while threshold_found == Int32(0) and iter < Int32(block_scan_iterations):
            bin_idx = Int32(num_bins - 1) - (iter * Int32(self.tb_size) + tid)
            count = hist_smem[bin_idx]
            chunk_inclusive = _block_scan_inclusive_i32(
                count,
                lane,
                warp_id,
                warp_totals_smem,
                self.warps_per_block,
            )
            running_count = running_count_smem.load()
            prior_in_scan_slice = chunk_inclusive - count
            remaining = Int32(self.topk) - running_count - prior_in_scan_slice
            if count > Int32(0) and remaining > Int32(0) and remaining <= count:
                threshold_bin_smem.store(bin_idx)
                if count <= remaining or cutlass.const_expr(is_final_pass):
                    include_threshold_bin_smem.store(Int32(1))
                threshold_found_smem.store(Int32(1))
            # Barrier: every thread must finish reading running_count for this
            # slice before tb_size-1 advances it, else a warp racing ahead to
            # the store makes a lagging thread double-count the slice total
            # (-> remaining too small -> threshold too high -> under-fill).
            cute.arch.sync_threads()
            if tid == Int32(self.tb_size - 1):
                running_count_smem.store(running_count + chunk_inclusive)
            cute.arch.sync_threads()

            threshold_found = threshold_found_smem.load()
            iter += Int32(1)

        threshold = threshold_bin_smem.load()
        should_include_threshold = include_threshold_bin_smem.load() != Int32(0)
        self._commit_keys(
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
        )
        cute.arch.sync_threads()

        pass_finished = include_threshold_bin_smem.load()
        if tid == Int32(0) and pass_finished == Int32(0):
            prefix_smem.store(prefix | (Uint64(threshold) << Uint64(shift)))
        cute.arch.sync_threads()
        return pass_finished

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
        committed_count_smem = storage.committed_count.data_ptr()
        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            key = keys[key_idx]
            if self._prefix_matches(key, prefix, prefix_bits):
                bin_idx = Int32((key >> Uint64(shift)) & bin_mask)
                selected = bin_idx > threshold
                if should_include_threshold:
                    selected = selected or bin_idx == threshold
                if selected:
                    dst = cute.arch.atomic_add(
                        committed_count_smem,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if dst < Int32(self.topk):
                        output[dst] = (~Uint32(key)).bitcast(Int32)

    @cute.kernel
    def kernel(
        self,
        input: cute.Pointer,
        out: cute.Pointer,
        rank_stride: Int64,
        out_stride: Int64,
    ):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = Int64(row)
        output_row = cute.make_tensor(
            out + row * out_stride,
            cute.make_layout((self.topk,)),
        )
        keys = cute.make_rmem_tensor((self.keys_per_thread,), Uint64)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage, 8)
        committed_count_smem = storage.committed_count.data_ptr()
        prefix_smem = storage.prefix_s.data_ptr()
        for i in range(tid, self.topk, self.tb_size):
            output_row[i] = Int32(-1)

        for key_idx in cutlass.range_constexpr(self.keys_per_thread):
            col = tid + Int32(key_idx * self.tb_size)
            rank = Int64(col // self.topk)
            candidate = col % self.topk
            offset = (
                rank * rank_stride + row * Int64(self.topk * 2) + Int64(candidate * 2)
            )
            score = (input + offset).load()
            token_id = Int32((input + offset + Int64(1)).load())
            keys[key_idx] = self._stable_key(score, token_id)

        if tid == Int32(0):
            committed_count_smem.store(Int32(0))
            prefix_smem.store(Uint64(0))
        cute.arch.sync_threads()

        step = Int32(0)
        finished = Int32(0)
        while finished == Int32(0) and step < Int32(self.radix_passes - 1):
            finished = self._radix_pass(
                keys,
                output_row,
                storage,
                tid,
                step,
                self.radix_bits,
                False,
            )
            step += Int32(1)

        if finished == Int32(0):
            self._radix_pass(
                keys,
                output_row,
                storage,
                tid,
                Int32(self.radix_passes - 1),
                self.final_radix_bits,
                True,
            )


@cache
def _compile(topk: int, world_size: int, device_index: int):
    raise_if_kernel_resolution_frozen(
        "cute.compile",
        target=_RankMajorTopKKernel,
        cache_key=(topk, world_size, device_index),
    )
    with torch.cuda.device(device_index):
        return b12x_compile(
            _RankMajorTopKKernel(topk, world_size),
            make_ptr(Float32, 16, cute.AddressSpace.gmem, assumed_align=4),
            make_ptr(Int32, 16, cute.AddressSpace.gmem, assumed_align=4),
            1,
            1,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "comm.pcie.rank_major_topk",
                2,
                topk,
                world_size,
                device_index,
            ),
        )


def precompile_rank_major_topk(topk: int, world_size: int, device) -> None:
    """Resolve static candidate geometry before graph capture or freezing."""
    if topk not in (512, 1024, 2048) or world_size not in (2, 4, 8):
        raise ValueError("Rank-major top-k requires K=512/1024/2048 and 2/4/8 ranks")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Rank-major top-k requires a CUDA device")
    index = torch.cuda.current_device() if device.index is None else device.index
    _compile(topk, world_size, index)


def rank_major_topk(gathered: torch.Tensor, out: torch.Tensor) -> None:
    """Write the selected token ids into caller-owned output storage."""
    if gathered.ndim != 4 or gathered.shape[-1] != 2:
        raise ValueError("Candidates must have shape [rank, row, K, 2]")
    world_size, rows, topk, _ = gathered.shape
    if (
        gathered.dtype != torch.float32
        or out.dtype != torch.int32
        or not gathered.is_cuda
        or out.device != gathered.device
        or out.shape != (rows, topk)
        or out.stride(1) != 1
        or (rows > 1 and out.stride(0) < topk)
        or gathered.stride(0) < rows * topk * 2
        or gathered.stride()[1:] != (topk * 2, 2, 1)
    ):
        raise ValueError(
            "Rank-major top-k tensor dtype, device, shape or stride mismatch"
        )
    precompile_rank_major_topk(topk, world_size, gathered.device)
    if rows:
        with torch.cuda.device(gathered.device):
            _compile(topk, world_size, gathered.device.index)(
                make_ptr(
                    Float32,
                    gathered.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                ),
                make_ptr(
                    Int32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
                ),
                rows,
                gathered.stride(0),
                out.stride(0),
                current_cuda_stream(),
            )


@triton.jit
def _pack_dcp_candidates_kernel(
    indices,
    scores,
    packed,
    index_stride,
    score_stride,
    packed_row_stride,
    packed_col_stride,
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    interleave: tl.constexpr,
    topk: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * block + tl.arange(0, block)
    mask = col < topk
    local_idx = tl.load(indices + row * index_stride + col, mask=mask, other=-1)
    score = tl.load(scores + row * score_stride + col, mask=mask, other=-float("inf"))
    valid = local_idx >= 0
    safe_idx = tl.maximum(local_idx, 0)
    global_idx = (
        (safe_idx // interleave) * (dcp_world_size * interleave)
        + dcp_rank * interleave
        + safe_idx % interleave
    )
    global_idx = tl.where(valid, global_idx, -1)
    score = tl.where(valid, score, -float("inf"))
    base = packed + row * packed_row_stride + col * packed_col_stride
    tl.store(base, score, mask=mask)
    tl.store(base + 1, global_idx.to(tl.float32), mask=mask)


def pack_dcp_candidates(indices, scores, packed, rank, world_size, interleave):
    """Publish local candidates as float32 (score, global token id) pairs."""
    topk = indices.shape[1]
    _pack_dcp_candidates_kernel[(indices.shape[0], triton.cdiv(topk, 512))](
        indices,
        scores,
        packed,
        indices.stride(0),
        scores.stride(0),
        packed.stride(0),
        packed.stride(1),
        rank,
        world_size,
        interleave,
        topk,
        512,
        num_warps=8,
    )
