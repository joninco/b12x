"""Caller-owned, strided BF16 GLM query projection with fixed MMA geometry."""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int64

from b12x._lib.compiler import KernelCompileSpec, compile as compile_kernel
from b12x._lib.gating import default_is_supported
from b12x._lib.utils import current_cuda_stream, make_ptr

from . import META
from ._kernel import QueryBmmKernel

_LAUNCHES: dict[int, object] = {}


def can_implement(
    *, num_heads: int, max_m: int, k: int = 192, n: int = 512, device=None
) -> bool:
    """Return whether the static geometry and device have a kernel implementation."""
    return (
        num_heads == 8
        and 1 <= max_m <= 8192
        and (k, n) == (192, 512)
        and is_supported(device)
    )


def is_supported(device=None) -> bool:
    """True when an SM120/SM121 target and the CUTLASS DSL are available."""
    return default_is_supported(device, requires=META.requires)


def _span(tensor: torch.Tensor) -> tuple[int, int]:
    elements = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.data_ptr(), tensor.data_ptr() + elements * tensor.element_size()


def _validate(lhs: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    if lhs.ndim != 3 or not 1 <= lhs.shape[1] <= 8192:
        raise ValueError("MLA query BMM requires 1..8192 rows in a rank-three query")
    rows = lhs.shape[1]
    for name, tensor, shape in [
        ("query", lhs, (8, rows, 192)),
        ("weight", weight, (8, 192, 512)),
        ("output", out, (8, rows, 512)),
    ]:
        if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
            raise ValueError(f"MLA query BMM {name} must be BF16 {shape}")
        if tensor.device != lhs.device or any(s <= 0 for s in tensor.stride()):
            raise ValueError(
                "MLA query BMM tensors require one device and positive strides"
            )
        span = 1
        for size, stride in sorted(
            zip(tensor.shape, tensor.stride(), strict=True), key=lambda entry: entry[1]
        ):
            if size > 1 and stride < span:
                raise ValueError(f"MLA query BMM {name} has overlapping dimensions")
            span += (size - 1) * stride
    if lhs.stride(2) != 1 or out.stride(2) != 1:
        raise ValueError("MLA query BMM query/output feature strides must be one")
    if out.data_ptr() % 4 or out.stride(0) % 2 or out.stride(1) % 2:
        raise ValueError("MLA query BMM output requires aligned BF16 pair stores")
    output_begin, output_end = _span(out)
    for tensor in (lhs, weight):
        begin, end = _span(tensor)
        if begin < output_end and output_begin < end:
            raise ValueError(
                "MLA query BMM output must not overlap an input storage span"
            )
    if lhs.device.type != "cuda":
        raise ValueError("MLA query BMM requires CUDA tensors")


def _get_launch(device: torch.device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    launch = _LAUNCHES.get(index)
    if launch is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MLA query BMM must be prewarmed before CUDA graph capture"
            )
        pointer = make_ptr(
            cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=2
        )
        output = make_ptr(cutlass.BFloat16, 16, cute.AddressSpace.gmem, assumed_align=4)
        launch = compile_kernel(
            QueryBmmKernel(),
            pointer,
            pointer,
            output,
            *(Int64(1) for _ in range(8)),
            current_cuda_stream(),
            compile_spec=KernelCompileSpec.from_facts(
                "mla_query_bmm_bf16",
                1,
                ("device_index", index),
                ("heads", 8),
                ("k", 192),
                ("n", 512),
                ("tile", (16, 32, 16)),
                ("operands", "bf16"),
                ("accumulator", "fp32"),
            ),
        )
        _LAUNCHES[index] = launch
    return launch


def run(lhs: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Compute BF16 query @ BF16 weight into supplied strided BF16 output.

    Shapes are [8,rows,192], [8,192,512], and [8,rows,512]. Weight K/N strides
    may be transposed; query/output feature strides must be one. No operand
    copies or temporary tensor allocations are performed. FP32 MMA reduction
    is rounded to BF16 once before the caller assembles the rotary tail.
    """
    _validate(lhs, weight, out)
    launch = _get_launch(lhs.device)
    launch(
        make_ptr(
            cutlass.BFloat16, lhs.data_ptr(), cute.AddressSpace.gmem, assumed_align=2
        ),
        make_ptr(
            cutlass.BFloat16, weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=2
        ),
        make_ptr(
            cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        ),
        Int64(lhs.shape[1]),
        Int64(lhs.stride(0)),
        Int64(lhs.stride(1)),
        Int64(weight.stride(0)),
        Int64(weight.stride(1)),
        Int64(weight.stride(2)),
        Int64(out.stride(0)),
        Int64(out.stride(1)),
        current_cuda_stream(),
    )
    return out


def prewarm(lhs: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Compile and first-launch using caller-provided buffers before capture."""
    run(lhs, weight, out)


def clear_caches() -> None:
    """Drop the compiled launches; the next call outside capture recompiles."""
    _LAUNCHES.clear()


__all__ = list(META.entry_points)
