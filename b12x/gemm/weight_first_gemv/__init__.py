"""Weight-first BF16 projection for decode rows that follow a PCIe one-shot
allreduce.

``y_i = x @ w_i.T`` for one or two BF16 weights ``w_i (N_i, K)`` that share
the activation ``x (M, K)``, ``M <= 16``. The weights are held as one
concatenated ``(N, K)`` buffer cut into bricks of ``NT x KT`` elements; one
CTA of 256 threads per brick stages its brick into shared memory with
``cp.async`` before ``griddepcontrol.wait``, so that with the
programmatic-stream-serialization launch attribute the weight read overlaps
the kernel that produces ``x`` (the fused one-shot allreduce, whose two
kernels trigger ``griddepcontrol.launch_dependents`` first). Products use
``mma.sync m16n8k16`` (bf16 in, fp32 accumulate); the last CTA of each
n-tile sums the ``K / KT`` fp32 split partials in split order, so the result
is bitwise repeatable.

``WeightFirstProjection`` owns the concatenated weight and the workspaces
and dispatches through the opaque custom op ``b12x::weight_first_gemv``
(torch.compile- and CUDA-graph-safe; shapes outside the contract fall back to
cuBLAS inside the op). ``precompile`` compiles and warm-runs the kernel for a
weight geometry at load time so serving never compiles inside a capture.

Example:
    from b12x.gemm import weight_first_gemv

    proj = weight_first_gemv.WeightFirstProjection([gate.weight, gate_up.weight])
    router_logits, gate_up_act = proj(hidden_states)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="weight_first_gemv",
    group="gemm",
    api_style="oneshot",
    entry_points=(
        "WeightFirstProjection",
        "weight_first_gemv",
        "brick_for",
        "supports",
        "precompile",
        "is_supported",
        "is_disabled",
        "MAX_ROWS",
        "DEFAULT_STAGE_DEPTH",
    ),
    dtypes=("bf16",),
    # Port of the production CUDA-extension kernel in the GLM-5.3 decode
    # profiling workspace (a non-git directory): ledger row H1, deployed
    # 2026-09-04 in the overlay tagged
    # a10a15b12b10cm16c8b15e9b18b19b10de10d1dc9c10g1h1s1a2.
    provenance=Provenance(
        repo="file:///home/jon/git/vllm-decode-profiling",
        commit="overlay-g1h1s1a2-2026-09-04",
        paths=("overlay/fork/wf_gemv.py", "kernel-bench/pdl_ext.py"),
    ),
    test_path="tests/gemm/test_weight_first_gemv.py",
    since="1.3.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        DEFAULT_STAGE_DEPTH,
        MAX_ROWS,
        WeightFirstProjection,
        brick_for,
        is_disabled,
        is_supported,
        precompile,
        supports,
        weight_first_gemv,
    )

install_lazy_api(globals(), META)
