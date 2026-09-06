"""Tests for the weight-first BF16 projection (``b12x::weight_first_gemv``).

Contract under test: for BF16 ``x (M, K)`` with ``1 <= M <= 16`` and one or
two BF16 weights ``(N_i, K)`` sharing ``K``, the op returns ``x @ w_i.T`` in
BF16 with fp32 accumulation in a reduction order fixed by the geometry alone
(bitwise repeatable across launches, row counts, staging depths and the
launch attribute); the compiled callable for a geometry serves every row
count without a new compile; rows beyond 16 take the cuBLAS fallback inside
the op; and the op captures into a CUDA graph and replays without host work
or allocation.
"""

from __future__ import annotations

import math

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _api():
    from b12x.gemm import weight_first_gemv

    weight_first_gemv.weight_first_gemv  # noqa: B018  (registers the op)
    return weight_first_gemv


def _random_weights(n_parts, k, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        (torch.randn(n, k, generator=g) * 0.05).to(torch.bfloat16).to(device)
        for n in n_parts
    ]


def _random_x(m, k, device, seed=1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(m, k, generator=g) * 0.5).to(torch.bfloat16).to(device)


def _assert_within_bf16_rounding(y: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    """``y`` must equal the float64 product to within BF16 rounding of the
    fp32-accumulated value: one BF16 ulp (2^-8 relative) plus the fp32
    accumulation noise of a K-term sum (K * 2^-24 relative to the L1 mass)."""
    ref = x.double() @ w.double().t()
    assert y.dtype == torch.bfloat16
    assert y.shape == (x.shape[0], w.shape[0])
    k = x.shape[1]
    mass = x.double().abs() @ w.double().abs().t()
    tol = ref.abs() * 2.0**-8 + mass * k * 2.0**-24 + 1e-6
    err = (y.double() - ref).abs()
    assert bool((err <= tol).all()), (
        f"max err {err.max().item():.3e} exceeds tolerance at "
        f"{(err - tol).argmax().item()}"
    )


# ----------------------------------------------------------------------------
# Host-side contract (no GPU needed)
# ----------------------------------------------------------------------------


def test_brick_selection():
    api = _api()
    assert api.brick_for(768, 6144) == (48, 768)
    assert api.brick_for(256, 6144) == (32, 768)
    assert api.brick_for(2048, 1536) == (48, 768)  # 1536 = 2 x 768
    assert api.brick_for(2624, 6144) == (48, 768)
    assert api.brick_for(64, 1024) == (64, 512)
    assert api.brick_for(64, 256) == (128, 256)
    with pytest.raises(ValueError):
        api.brick_for(64, 100)


def test_registry_lists_op():
    import b12x
    from b12x import gemm

    assert "gemm.weight_first_gemv" in b12x._OPS
    assert "weight_first_gemv" in gemm._OP_MODULES
    api = _api()
    assert api.MAX_ROWS == 16
    assert api.DEFAULT_STAGE_DEPTH == 1


def test_supports_rejects_out_of_contract_inputs():
    api = _api()
    w_cpu = torch.zeros(64, 768, dtype=torch.bfloat16)
    assert not api.supports([w_cpu])
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    w = torch.zeros(64, 768, dtype=torch.bfloat16, device=device)
    assert api.supports([w])
    assert api.supports([w, torch.zeros(32, 768, dtype=torch.bfloat16, device=device)])
    assert not api.supports([])
    assert not api.supports([w, w, w])
    assert not api.supports([w.float()])
    assert not api.supports([torch.zeros(64, 100, dtype=torch.bfloat16, device=device)])
    assert not api.supports(
        [w, torch.zeros(32, 512, dtype=torch.bfloat16, device=device)]
    )
    assert not api.supports([w.t().contiguous().t()])  # non-contiguous
    x = torch.zeros(4, 768, dtype=torch.bfloat16, device=device)
    assert api.supports([w], x)
    assert not api.supports(
        [w], torch.zeros(17, 768, dtype=torch.bfloat16, device=device)
    )
    assert not api.supports(
        [w], torch.zeros(4, 512, dtype=torch.bfloat16, device=device)
    )
    assert not api.supports([w], x.float())


def test_disabled_switch(monkeypatch):
    api = _api()
    monkeypatch.setenv("B12X_DISABLE_WEIGHT_FIRST_GEMV", "1")
    assert api.is_disabled()
    assert not api.is_supported()
    monkeypatch.delenv("B12X_DISABLE_WEIGHT_FIRST_GEMV")
    assert not api.is_disabled()


# ----------------------------------------------------------------------------
# GPU contract
# ----------------------------------------------------------------------------

GEOMETRIES = [
    # (n_parts, k): production router + shared gate_up geometry first.
    ((256, 512), 6144),  # brick 48x768, 8 splits, two outputs
    ((256,), 6144),  # brick 32x768, 8 splits
    ((100,), 768),  # partial n-tile, single split (direct bf16 store)
    ((48, 52), 1536),  # two splits: remainder path of the split reduction
    ((96,), 2304),  # three splits
    ((200,), 3072),  # four splits: quad path only
    ((64, 64), 1024),  # brick 64x512, two splits
    ((128,), 256),  # brick 128x256, single split
    ((2048,), 1536),  # q_b-like: brick 48x768, two splits, 43 n-tiles
]


@cuda_required
@pytest.mark.parametrize("n_parts,k", GEOMETRIES)
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_matches_float64_reference(n_parts, k, m):
    api = _api()
    device = torch.device("cuda")
    weights = _random_weights(n_parts, k, device)
    cat = torch.cat(weights, dim=0)
    proj = api.WeightFirstProjection(weights)
    x = _random_x(m, k, device)
    outs = proj(x)
    assert len(outs) == len(n_parts)
    offset = 0
    for y, n in zip(outs, n_parts, strict=True):
        _assert_within_bf16_rounding(y, x, cat[offset : offset + n])
        offset += n


@cuda_required
def test_parameters_become_views_into_concatenated_buffer():
    api = _api()
    device = torch.device("cuda")
    weights = _random_weights((256, 512), 6144, device)
    originals = [w.clone() for w in weights]
    proj = api.WeightFirstProjection(weights)
    assert proj.weight.is_contiguous()
    assert proj.weight.shape == (768, 6144)
    for w, orig in zip(weights, originals, strict=True):
        assert torch.equal(w, orig)
        assert w.data_ptr() >= proj.weight.data_ptr()
        assert w.data_ptr() < proj.weight.data_ptr() + proj.weight.numel() * 2
        assert w.is_contiguous()
    assert weights[1].data_ptr() == weights[0].data_ptr() + 256 * 6144 * 2
    # In-place edits to a parameter reach the kernel through the shared storage.
    weights[0].zero_()
    x = _random_x(4, 6144, device)
    y0, y1 = proj(x)
    assert torch.count_nonzero(y0) == 0
    _assert_within_bf16_rounding(y1, x, originals[1])


@cuda_required
def test_bitwise_repeatable_across_launches_depths_and_attribute():
    api = _api()
    device = torch.device("cuda")
    weights = _random_weights((256, 512), 6144, device)
    proj = api.WeightFirstProjection(weights)
    x = _random_x(8, 6144, device)
    first = [y.clone() for y in proj(x)]
    for _ in range(30):
        for y, ref in zip(proj(x), first, strict=True):
            assert torch.equal(y, ref)
    for depth in (0, 1, 2, 3, 4, 6, 8):
        for pdl in (False, True):
            proj.depth = depth
            proj.pdl = pdl
            for y, ref in zip(proj(x), first, strict=True):
                assert torch.equal(y, ref), (depth, pdl)


@cuda_required
def test_rows_independent_of_batch():
    """Row ``i`` of a 16-row launch equals the single-row launch of that row
    (the reduction order does not depend on M)."""
    api = _api()
    device = torch.device("cuda")
    weights = _random_weights((256, 512), 6144, device)
    proj = api.WeightFirstProjection(weights)
    x = _random_x(16, 6144, device)
    full = proj(x)
    for i in range(16):
        single = proj(x[i : i + 1].contiguous())
        for ys, yf in zip(single, full, strict=True):
            assert torch.equal(ys[0], yf[i]), i


@cuda_required
def test_all_row_counts_under_frozen_resolution():
    """After ``precompile`` every row count 1..16 runs under frozen kernel
    resolution (no compile, no cache lookup by M)."""
    import b12x

    api = _api()
    device = torch.device("cuda")
    weights = _random_weights((256, 512), 6144, device)
    proj = api.WeightFirstProjection(weights)  # precompiles
    assert proj.compiled
    cat = proj.weight.clone()
    b12x.freeze_kernel_resolution("weight-first projection row-count test")
    try:
        for m in range(1, 17):
            x = _random_x(m, 6144, device, seed=100 + m)
            y0, y1 = proj(x)
            _assert_within_bf16_rounding(y0, x, cat[:256])
            _assert_within_bf16_rounding(y1, x, cat[256:])
    finally:
        b12x.unfreeze_kernel_resolution()


@cuda_required
def test_rows_beyond_max_fall_back_to_cublas():
    api = _api()
    device = torch.device("cuda")
    weights = _random_weights((256, 512), 6144, device)
    proj = api.WeightFirstProjection(weights)
    cat = proj.weight.clone()
    x = _random_x(17, 6144, device)
    assert not proj.supports(x)
    y0, y1 = proj(x)
    ref = torch.nn.functional.linear(x, cat)
    assert torch.equal(y0, ref[:, :256])
    assert torch.equal(y1, ref[:, 256:])


@cuda_required
def test_graph_capture_replay_without_allocation():
    """The op captures into a CUDA graph; replays reproduce the eager result
    bitwise on new inputs and neither allocate nor compile."""
    import b12x

    api = _api()
    device = torch.device("cuda")
    weights = _random_weights((256, 512), 6144, device)
    proj = api.WeightFirstProjection(weights)
    static_x = _random_x(4, 6144, device).clone()
    eager = [y.clone() for y in proj(static_x)]
    stream = torch.cuda.Stream(device)
    graph = torch.cuda.CUDAGraph()
    b12x.freeze_kernel_resolution("weight-first projection capture test")
    try:
        with torch.cuda.stream(stream):
            proj(static_x)  # warm the allocator on the side stream
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=stream):
                out = proj(static_x)
        torch.cuda.synchronize(device)
        graph.replay()
        torch.cuda.synchronize(device)
        for y, ref in zip(out, eager, strict=True):
            assert torch.equal(y, ref)
        allocated = torch.cuda.memory_allocated(device)
        new_x = _random_x(4, 6144, device, seed=7)
        static_x.copy_(new_x)
        expected = [y.clone() for y in proj(new_x)]
        allocated = torch.cuda.memory_allocated(device)
        for _ in range(20):
            graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_allocated(device) == allocated
        for y, ref in zip(out, expected, strict=True):
            assert torch.equal(y, ref)
    finally:
        b12x.unfreeze_kernel_resolution()


@cuda_required
def test_direct_op_single_weight_and_zero_second_output():
    api = _api()
    device = torch.device("cuda")
    (w,) = _random_weights((320,), 6144, device)
    proj = api.WeightFirstProjection([w])
    x = _random_x(3, 6144, device)
    (y,) = proj(x)
    _assert_within_bf16_rounding(y, x, w)
    y0, y1 = torch.ops.b12x.weight_first_gemv(
        x, proj.weight, proj.partial, proj.counters, 320, proj.nt, proj.kt, 1, True
    )
    assert torch.equal(y0, y)
    assert y1.shape == (3, 0)


@cuda_required
def test_counters_return_to_zero():
    """The last CTA of each n-tile resets its arrival counter, so a launch
    leaves the workspace ready for the next one."""
    api = _api()
    device = torch.device("cuda")
    weights = _random_weights((256, 512), 6144, device)
    proj = api.WeightFirstProjection(weights)
    for m in (1, 16):
        proj(_random_x(m, 6144, device))
        torch.cuda.synchronize(device)
        assert torch.count_nonzero(proj.counters) == 0
    assert proj.counters.numel() == math.ceil(768 / 48)
