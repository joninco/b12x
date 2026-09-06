"""Public surface for attention.topk_sort (docs in the op ``__init__``)."""

from __future__ import annotations

import os

import torch

from ..._lib.gating import default_is_supported
from . import META
from ._kernel import MAX_BITMAP_WORDS, bitmap_words  # noqa: F401
from ._kernel import compile_sort_convert
from ._kernel import sort_convert  # noqa: F401  (registers the op; alias)

_PRECOMPILED: set[int] = set()


def is_disabled() -> bool:
    """True when ``B12X_DISABLE_TOPK_SORT`` turns the op off (debug
    isolation switch)."""
    return os.environ.get("B12X_DISABLE_TOPK_SORT", "").lower() in (
        "1",
        "true",
        "yes",
    )


def is_supported(device=None) -> bool:
    """True on SM120/SM121 with the required CUTLASS DSL, unless disabled via
    ``B12X_DISABLE_TOPK_SORT``."""
    if is_disabled():
        return False
    return default_is_supported(device, requires=META.requires)


def supports(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> bool:
    """Whether the kernel serves these inputs: CUDA int32 ``indices [rows,
    topk]`` with contiguous rows, contiguous int32 ``seq_lens [>= rows]``,
    int32 ``block_table [>= rows, width]`` on the same device, a power-of-two
    ``block_size`` and ``max_positions`` within the bitmap limit."""
    try:
        from ._kernel import _check_inputs

        _check_inputs(indices, seq_lens, block_table, block_size)
        bitmap_words(max_positions)
    except ValueError:
        return False
    return True


def precompile(max_positions: int, device: torch.device, log=None) -> int:
    """Compile and warm-run the sort for ``max_positions`` so a later launch
    inside a CUDA-graph capture neither compiles nor loads a module. Returns
    the bitmap word count (the compile key)."""
    if log is None:
        import logging

        log = logging.getLogger("b12x.topk_sort")
    words = bitmap_words(max_positions)
    if words in _PRECOMPILED:
        return words
    log.info(
        "topk_sort precompile: max_positions=%d bitmap words=%d", max_positions, words
    )
    launch = compile_sort_convert(words)
    indices = torch.full((2, 64), -1, dtype=torch.int32, device=device)
    indices[:, :4] = torch.tensor([3, 1, 2, 0], dtype=torch.int32, device=device)
    seq_lens = torch.full((2,), 64, dtype=torch.int32, device=device)
    block_table = torch.arange(2 * 4, dtype=torch.int32, device=device).view(2, 4)
    launch(indices, seq_lens, block_table, 64)
    torch.cuda.synchronize(device)
    _PRECOMPILED.add(words)
    return words


def sort_convert_reference(
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    max_positions: int,
) -> torch.Tensor:
    """Pure-torch semantics of ``sort_convert`` (returns a new tensor)."""
    rows, topk = indices.shape
    words = bitmap_words(max_positions)
    out = torch.full_like(indices, -1)
    log2 = int(block_size).bit_length() - 1
    for row in range(rows):
        limit = min(max(int(seq_lens[row]), 0), words * 32)
        limit = min((limit + 31) // 32 * 32, words * 32)
        positions = indices[row].to(torch.int64)
        positions = positions[(positions >= 0) & (positions < limit)]
        positions = torch.unique(positions, sorted=True)
        blocks = positions >> log2
        width = int(block_table.shape[1])
        pages = torch.where(
            blocks < width,
            block_table[row].to(torch.int64)[blocks.clamp(max=width - 1)],
            torch.full_like(blocks, -1),
        )
        slots = torch.where(
            blocks < width,
            (pages << log2) | (positions & (block_size - 1)),
            torch.full_like(blocks, -1),
        )
        count = min(int(slots.numel()), topk)
        out[row, :count] = slots[:count].to(torch.int32)
    return out


__all__ = [
    "sort_convert",
    "sort_convert_reference",
    "precompile",
    "supports",
    "is_supported",
    "is_disabled",
    "bitmap_words",
    "MAX_BITMAP_WORDS",
]
