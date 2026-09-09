"""Mask natural-log partial LSE with per-query local KV lengths."""

from functools import cache

import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr


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
