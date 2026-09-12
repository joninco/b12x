"""Strided BF16 GLM query BMM with caller-owned output and FP32 accumulation.

The one-shot operation multiplies an eight-head BF16 query of 192 features by a
BF16 ``[8, 192, 512]`` weight into a caller-owned strided BF16 output. Row
counts and operand strides are runtime values of one compiled kernel, weights
stay BF16, and the rotary tail is neither allocated nor assembled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="mla_query_bmm",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "run",
        "prewarm",
        "can_implement",
        "is_supported",
        "clear_caches",
    ),
    dtypes=("bf16",),
    recipes=("bf16",),
    requires=("cutlass",),
    provenance=Provenance(
        repo="https://github.com/local-inference-lab/b12x",
        commit="e25f3fbf",
        paths=("b12x/gemm/mla_query_bmm/",),
    ),
    test_path="tests/gemm/test_mla_query_bmm.py",
    since="1.4.0",
    notes="Caller-owned strided BF16 query BMM with FP32 MMA accumulation.",
)

if TYPE_CHECKING:
    from .api import (  # noqa: F401
        can_implement,
        clear_caches,
        is_supported,
        prewarm,
        run,
    )

install_lazy_api(globals(), META)
