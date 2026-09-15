"""Stable selected-token mapping into rank-major gathered native KV records."""

import torch
import triton
import triton.language as tl


# One program maps one query row over a single power-of-two tile, so the
# accepted width is bounded by the largest tile the scan handles in one
# program. 4096 covers the 2048-token DSA selection and the GLM5-Next
# selection with its pooled tail (2048 + 4 - 1 columns).
_MAX_WIDTH = 4096


@triton.jit(
    do_not_specialize=[
        "requests",
        "req_stride",
        "starts_stride0",
        "starts_stride1",
        "lens_stride0",
        "lens_stride1",
        "ti_stride0",
        "ti_stride1",
        "out_stride0",
        "out_stride1",
        "count_stride",
        "padded_rank_tokens",
    ]
)
def _map_global_topk_to_gathered_ckv_kernel(
    req_ids,
    token_indices,
    rank_starts,
    rank_lengths,
    output,
    counts,
    requests,
    req_stride,
    starts_stride0,
    starts_stride1,
    lens_stride0,
    lens_stride1,
    ti_stride0,
    ti_stride1,
    out_stride0,
    out_stride1,
    count_stride,
    padded_rank_tokens,
    DCP_SIZE: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK).to(tl.int64)
    request = tl.load(req_ids + row * req_stride).to(tl.int64)
    token = tl.load(
        token_indices + row * ti_stride0 + col * ti_stride1, col < TOPK, other=-1
    ).to(tl.int64)
    owner = (token // INTERLEAVE) % DCP_SIZE
    local = token // (DCP_SIZE * INTERLEAVE) * INTERLEAVE + token % INTERLEAVE
    readable = (col < TOPK) & (token >= 0) & (request >= 0) & (request < requests)
    start = tl.load(
        rank_starts + owner * starts_stride0 + request * starts_stride1,
        readable,
        other=0,
    ).to(tl.int64)
    length = tl.load(
        rank_lengths + owner * lens_stride0 + request * lens_stride1, readable, other=0
    ).to(tl.int64)
    valid = (
        readable
        & (local >= 0)
        & (local < length)
        & (start >= 0)
        & (start + local < padded_rank_tokens)
    )
    flags = valid.to(tl.int32)
    offset = (tl.cumsum(flags) - flags).to(tl.int64)
    slot = owner * padded_rank_tokens + start + local
    tl.store(
        output + row * out_stride0 + offset * out_stride1, slot.to(tl.int32), valid
    )
    tl.store(counts + row * count_stride, tl.sum(flags))


def _byte_span(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.numel() == 0:
        return tensor.data_ptr(), tensor.data_ptr()
    size = 1 + sum(
        (length - 1) * stride
        for length, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    return tensor.data_ptr(), tensor.data_ptr() + size * tensor.element_size()


def _nonoverlapping(tensor: torch.Tensor) -> bool:
    span = 1
    for size, stride in sorted(
        zip(tensor.shape, tensor.stride(), strict=True), key=lambda item: item[1]
    ):
        if size > 1 and stride < span:
            return False
        span += (size - 1) * stride
    return True


def map_global_topk_to_gathered_ckv(
    req_ids: torch.Tensor,
    token_indices: torch.Tensor,
    rank_req_starts: torch.Tensor,
    rank_req_lens: torch.Tensor,
    out: torch.Tensor,
    valid_counts: torch.Tensor,
    *,
    dcp_size: int,
    cp_kv_cache_interleave_size: int,
    padded_rank_tokens: int,
) -> None:
    """Map and stably compact up to 4096 selected IDs per query row.

    Metadata and outputs are int32; pointer arithmetic is int64. Invalid request
    IDs, negative tokens, missing local positions and rank-padding addresses are
    omitted. Surviving entries, including duplicates, retain input order. Counts
    are written directly and the unused output tail is -1. Output initialization
    is a separate ordered operation so scattered stores cannot race a tail fill.
    """
    tensors = (
        req_ids,
        token_indices,
        rank_req_starts,
        rank_req_lens,
        out,
        valid_counts,
    )
    if any(t.dtype != torch.int32 for t in tensors):
        raise TypeError("CKV gather index metadata must be int32")
    if token_indices.ndim != 2 or not 1 <= token_indices.shape[1] <= _MAX_WIDTH:
        raise ValueError(
            f"CKV selected indices require a matrix with width 1..{_MAX_WIDTH}"
        )
    if out.shape != token_indices.shape:
        raise ValueError("CKV gather index output shape does not match top-k input")
    rows = token_indices.shape[0]
    if req_ids.shape != (rows,) or valid_counts.shape != (rows,):
        raise ValueError("CKV request IDs and counts must have one entry per query row")
    if (
        rank_req_starts.ndim != 2
        or rank_req_starts.shape != rank_req_lens.shape
        or rank_req_starts.shape[0] != dcp_size
    ):
        raise ValueError(
            "CKV request starts/lens must have matching (DCP, requests) shape"
        )
    if (
        dcp_size < 1
        or cp_kv_cache_interleave_size < 1
        or padded_rank_tokens < 0
        or dcp_size * padded_rank_tokens > 2**31
    ):
        raise ValueError("CKV rank geometry must fit non-negative int32 output slots")
    if any(t.device != out.device for t in tensors):
        raise ValueError("CKV mapping tensors must share one device")
    if any(any(s < 0 for s in t.stride()) for t in tensors):
        raise ValueError("CKV mapping requires non-negative tensor strides")
    if not _nonoverlapping(out) or not _nonoverlapping(valid_counts):
        raise ValueError("CKV outputs must not overlap themselves")
    inputs = (req_ids, token_indices, rank_req_starts, rank_req_lens)
    for destination, sources in (
        (out, (*inputs, valid_counts)),
        (valid_counts, inputs),
    ):
        begin, end = _byte_span(destination)
        for source in sources:
            other_begin, other_end = _byte_span(source)
            if begin < other_end and other_begin < end:
                raise ValueError("CKV output storage overlaps another tensor span")
    if out.device.type != "cuda":
        raise ValueError("CKV selected-token mapping requires CUDA tensors")
    out.fill_(-1)
    if rows:
        _map_global_topk_to_gathered_ckv_kernel[(rows,)](
            req_ids,
            token_indices,
            rank_req_starts,
            rank_req_lens,
            out,
            valid_counts,
            rank_req_starts.shape[1],
            req_ids.stride(0),
            rank_req_starts.stride(0),
            rank_req_starts.stride(1),
            rank_req_lens.stride(0),
            rank_req_lens.stride(1),
            token_indices.stride(0),
            token_indices.stride(1),
            out.stride(0),
            out.stride(1),
            valid_counts.stride(0),
            padded_rank_tokens,
            dcp_size,
            cp_kv_cache_interleave_size,
            token_indices.shape[1],
            triton.next_power_of_2(token_indices.shape[1]),
        )
