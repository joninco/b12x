"""Localize DCP causal lengths and mask natural-log partial LSE."""

from functools import cache

import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


class _DCPSequenceLengths:
    @cute.jit
    def __call__(
        self,
        lengths: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        world_size: Int32,
        rank: Int32,
        interleave: Int32,
        stream: CUstream,
    ):
        self.localize(lengths, out, rows, world_size, rank, interleave).launch(
            grid=((rows + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream
        )

    @cute.kernel
    def localize(
        self,
        lengths: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        world_size: Int32,
        rank: Int32,
        interleave: Int32,
    ):
        block, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        row = block * 128 + thread
        if row < rows:
            length = (lengths + Int64(row)).load()
            cycle = world_size * interleave
            rounds = length // cycle
            remainder = length - rounds * cycle - rank * interleave
            if remainder < Int32(0):
                remainder = Int32(0)
            if remainder > interleave:
                remainder = interleave
            (out + Int64(row)).store(rounds * interleave + remainder)


@cache
def precompile_dcp_sequence_lengths(device_index: int):
    """Compile one callable per device; live lengths and rows are runtime data."""
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=_DCPSequenceLengths, cache_key=(device_index,)
    )
    with torch.cuda.device(device_index):
        return b12x_compile(
            _DCPSequenceLengths(),
            _ptr(Int32, 16),
            _ptr(Int32, 16),
            1,
            1,
            0,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "comm.pcie.dcp_sequence_lengths", 1, device_index
            ),
        )


def localize_dcp_sequence_lengths(
    lengths: torch.Tensor,
    out: torch.Tensor,
    world_size: int,
    rank: int,
    interleave: int,
) -> None:
    """Write local counts for nonnegative global bounds, allowing exact aliasing."""
    if (
        lengths.ndim != 1
        or out.shape != lengths.shape
        or lengths.dtype != torch.int32
        or out.dtype != torch.int32
        or not lengths.is_cuda
        or out.device != lengths.device
        or not lengths.is_contiguous()
        or not out.is_contiguous()
        or world_size < 1
        or not 0 <= rank < world_size
        or interleave < 1
        or world_size * interleave > 2**31 - 1
        or (
            torch._C._overlaps(lengths, out)
            and lengths.data_ptr() != out.data_ptr()
        )
    ):
        raise ValueError(
            "DCP lengths require contiguous CUDA int32 vectors, valid shard "
            "geometry, and disjoint or exactly aliased storage"
        )
    if lengths.numel() == 0:
        return
    with torch.cuda.device(lengths.device):
        precompile_dcp_sequence_lengths(lengths.device.index)(
            _ptr(Int32, lengths.data_ptr()),
            _ptr(Int32, out.data_ptr()),
            lengths.numel(),
            world_size,
            rank,
            interleave,
            current_cuda_stream(),
        )


class _MaskLocalLSE:
    def __init__(self, heads):
        self.heads = heads

    @cute.jit
    def __call__(
        self,
        lse: cute.Pointer,
        lengths: cute.Pointer,
        out: cute.Pointer,
        rows: Int32,
        row_stride: Int64,
        head_stride: Int64,
        stream: CUstream,
    ):
        self.mask(lse, lengths, out, row_stride, head_stride).launch(
            grid=(rows, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def mask(
        self,
        lse: cute.Pointer,
        lengths: cute.Pointer,
        out: cute.Pointer,
        row_stride: Int64,
        head_stride: Int64,
    ):
        row, _, _ = cute.arch.block_idx()
        thread, _, _ = cute.arch.thread_idx()
        local_length = (lengths + Int64(row)).load()
        for head in range(thread, self.heads, 32):
            value = Float32(-float("inf"))
            if local_length > Int32(0):
                value = (
                    lse + Int64(row) * row_stride + Int64(head) * head_stride
                ).load()
            (out + Int64(row) * Int64(self.heads) + Int64(head)).store(value)


def _ptr(dtype, address):
    return make_ptr(dtype, address, cute.AddressSpace.gmem, assumed_align=4)


@cache
def precompile_local_lse_mask(heads: int, device_index: int):
    if heads <= 0:
        raise ValueError("DCP LSE head count must be positive")
    key = (heads, device_index)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=_MaskLocalLSE, cache_key=key
    )
    with torch.cuda.device(device_index):
        return b12x_compile(
            _MaskLocalLSE(heads),
            _ptr(Float32, 16),
            _ptr(Int32, 16),
            _ptr(Float32, 16),
            1,
            1,
            1,
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "comm.pcie.local_lse_mask", 1, *key
            ),
        )


def mask_local_lse(lse: torch.Tensor, lengths: torch.Tensor, out: torch.Tensor) -> None:
    if (
        lse.ndim != 2
        or lse.shape[0] < 1
        or out.shape != lse.shape
        or lengths.shape != lse.shape[:1]
        or lse.dtype != torch.float32
        or out.dtype != torch.float32
        or lengths.dtype != torch.int32
        or not lse.is_cuda
        or out.device != lse.device
        or lengths.device != lse.device
        or not out.is_contiguous()
        or not lengths.is_contiguous()
    ):
        raise ValueError(
            "DCP local LSE masking requires float32 LSE and int32 per-query lengths"
        )
    with torch.cuda.device(lse.device):
        precompile_local_lse_mask(lse.shape[1], lse.device.index)(
            _ptr(Float32, lse.data_ptr()),
            _ptr(Int32, lengths.data_ptr()),
            _ptr(Float32, out.data_ptr()),
            lse.shape[0],
            lse.stride(0),
            lse.stride(1),
            current_cuda_stream(),
        )
