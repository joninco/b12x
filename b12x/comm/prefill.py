"""Caller-owned LSE preparation for eager prefill output combination."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rows", "row_stride", "head_stride"])
def _prepare_prefill_lse_kernel(
    source,
    lengths,
    output,
    rows,
    row_stride,
    head_stride,
    HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row = offset // HEADS
    head = offset % HEADS
    valid = row < rows
    length = tl.load(lengths + row, valid, other=0)
    lse = tl.load(source + row * row_stride + head * head_stride, valid, other=0.0)
    keep = (length > 0) & (lse == lse) & (lse != float("inf"))
    tl.store(output + offset, tl.where(keep, lse, float("-inf")), valid)


def prepare_prefill_lse(
    lse: torch.Tensor,
    local_causal_lengths: torch.Tensor,
    out: torch.Tensor,
    *,
    num_rows: int | None = None,
) -> None:
    """Copy LSE into persistent scratch, masking empty or invalid partials.

    Zero/negative local lengths, NaN, and positive infinity become negative
    infinity. Finite LSE and negative infinity remain unchanged for nonempty
    shards. Only active output rows are written. Output must not overlap the
    source or lengths; it survives reuse of the source's borrowed workspace.
    """
    if (
        lse.ndim != 2
        or out.ndim != 2
        or lse.dtype != torch.float32
        or out.dtype != torch.float32
        or not out.is_contiguous()
        or lse.shape[1] != out.shape[1]
        or out.shape[1] < 1
        or any(stride <= 0 for stride in lse.stride())
    ):
        raise ValueError("Prefill LSE requires FP32 matrices and contiguous output")
    if (
        local_causal_lengths.ndim != 1
        or local_causal_lengths.dtype != torch.int32
        or not local_causal_lengths.is_contiguous()
    ):
        raise ValueError("Prefill causal lengths require a contiguous int32 vector")
    rows = out.shape[0] if num_rows is None else num_rows
    if not 0 <= rows <= min(lse.shape[0], out.shape[0], local_causal_lengths.numel()):
        raise ValueError("Prefill LSE live rows exceed input or output capacity")
    if any(t.device != out.device for t in (lse, local_causal_lengths)):
        raise ValueError("Prefill LSE tensors must share one device")
    output_begin = out.data_ptr()
    output_end = output_begin + rows * out.shape[1] * out.element_size()
    for source in (lse, local_causal_lengths):
        span = (
            sum(
                (size - 1) * stride
                for size, stride in zip(source.shape, source.stride(), strict=True)
            )
            + 1
        )
        source_begin = source.data_ptr()
        source_end = source_begin + span * source.element_size()
        if rows and source_begin < output_end and output_begin < source_end:
            raise ValueError("Prefill LSE output overlaps an input storage span")
    if out.device.type != "cuda":
        raise ValueError("Prefill LSE preparation requires CUDA tensors")
    if rows:
        _prepare_prefill_lse_kernel[(triton.cdiv(rows * out.shape[1], 256),)](
            lse,
            local_causal_lengths,
            out,
            rows,
            lse.stride(0),
            lse.stride(1),
            out.shape[1],
            256,
        )
