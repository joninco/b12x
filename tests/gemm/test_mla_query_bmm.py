"""Strided BF16 query-BMM contracts, FP64 oracle and fresh-input graph replay.

CPU validation catches invalid layouts and ownership; GPU tests establish MMA
numerics for the eight-head 192-to-512 projection without quantizing weights.
"""

import pytest
import torch

from b12x.gemm import mla_query_bmm
from b12x.gemm.mla_query_bmm import api
from tests._reference.helpers import require_b12x


def _inputs(rows, *, device="cpu", transposed_weight=False):
    packed = torch.empty((rows + 1, 8, 256), dtype=torch.bfloat16, device=device)
    query = packed[1:, :, :192].permute(1, 0, 2)
    if transposed_weight:
        weight = torch.empty(
            (8, 512, 192), dtype=torch.bfloat16, device=device
        ).transpose(1, 2)
    else:
        weight = torch.empty((8, 192, 520), dtype=torch.bfloat16, device=device)[
            :, :, :512
        ]
    storage = torch.full((rows + 2, 8, 576), -17.0, dtype=torch.bfloat16, device=device)
    out = storage[1 : rows + 1, :, :512].permute(1, 0, 2)
    return query, weight, out, storage


def _assert_fp32_accumulation_bound(query, weight, result):
    lhs, rhs = query.double(), weight.double()
    reference = torch.bmm(lhs, rhs)
    # FP32 dot-product forward-error bound plus one BF16 output rounding.
    u = torch.finfo(torch.float32).eps / 2
    gamma = 192 * u / (1 - 192 * u)
    accumulation = gamma * torch.bmm(lhs.abs(), rhs.abs())
    bound = (
        accumulation
        + (reference.abs() + accumulation) * torch.finfo(torch.bfloat16).eps / 2
    )
    error = (result.double() - reference).abs()
    assert torch.all(error <= bound + torch.finfo(torch.bfloat16).tiny)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({}, True),
        ({"num_heads": 16}, False),
        ({"max_m": 8193}, False),
        ({"max_m": 0}, False),
        ({"k": 512}, False),
    ],
)
def test_query_bmm_support_is_exact_geometry(monkeypatch, kwargs, expected):
    monkeypatch.setattr(api, "default_is_supported", lambda *a, **kw: True)
    args = dict(num_heads=8, max_m=8192)
    args.update(kwargs)
    assert mla_query_bmm.can_implement(**args) is expected


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "rows",
        "query_stride",
        "weight_shape",
        "overlap",
        "output_alignment",
        "device",
    ],
)
def test_query_bmm_rejects_invalid_contract_without_launch(case):
    query, weight, out, _ = _inputs(3)
    if case == "dtype":
        query = query.float()
    elif case == "rows":
        query = query[:, :0]
    elif case == "query_stride":
        query = torch.empty((8, 3, 384), dtype=torch.bfloat16)[:, :, ::2]
    elif case == "weight_shape":
        weight = weight[:, :, :256]
    elif case == "overlap":
        storage = torch.empty((8, 3, 512), dtype=torch.bfloat16)
        query, out = storage[:, :, :192], storage
    elif case == "output_alignment":
        storage = torch.empty(8 * 3 * 512 + 1, dtype=torch.bfloat16)
        out = storage[1:].view(8, 3, 512)
    with pytest.raises(ValueError):
        mla_query_bmm.run(query, weight, out)


@pytest.mark.parametrize("rows", [1, 17, 257, 8192])
@pytest.mark.parametrize("transposed_weight", [False, True])
def test_query_bmm_pitched_inputs_match_fp64_bound(rows, transposed_weight):
    require_b12x()
    query, weight, out, storage = _inputs(
        rows, device="cuda", transposed_weight=transposed_weight
    )
    for seed in [13, 71]:
        torch.manual_seed(seed)
        query.normal_()
        weight.normal_()
        assert mla_query_bmm.run(query, weight, out) is out
        _assert_fp32_accumulation_bound(query, weight, out)
        assert torch.all(storage[0] == -17)
        assert torch.all(storage[-1] == -17)
        assert torch.all(storage[1:-1, :, 512:] == -17)


def test_query_bmm_runtime_rows_and_strides_reuse_compilation(monkeypatch):
    require_b12x()
    query, weight, out, _ = _inputs(257, device="cuda")
    query.normal_()
    weight.normal_()
    mla_query_bmm.prewarm(query, weight, out)
    compiled_ids = {key: id(value) for key, value in api._LAUNCHES.items()}

    def reject_compile(*args, **kwargs):
        raise AssertionError("query BMM recompiled for live rows or physical strides")

    monkeypatch.setattr(api, "compile_kernel", reject_compile)
    for rows in [1, 17, 257]:
        q, w, output, _ = _inputs(rows, device="cuda", transposed_weight=True)
        q.normal_()
        w.normal_()
        mla_query_bmm.run(q, w, output)
        _assert_fp32_accumulation_bound(q, w, output)
    assert {key: id(value) for key, value in api._LAUNCHES.items()} == compiled_ids


def test_query_bmm_graph_replay_uses_changed_inputs_and_stable_output():
    require_b12x()
    query, weight, out, _ = _inputs(33, device="cuda", transposed_weight=True)
    query.normal_()
    weight.normal_()
    mla_query_bmm.prewarm(query, weight, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_query_bmm.run(query, weight, out)
    pointer = out.data_ptr()
    for seed in [3, 7]:
        torch.manual_seed(seed)
        query.normal_()
        weight.normal_()
        graph.replay()
        _assert_fp32_accumulation_bound(query, weight, out)
        assert out.data_ptr() == pointer


def test_query_bmm_is_registered_with_lazy_entry_points():
    import b12x

    meta = b12x.find_op("gemm.mla_query_bmm")
    assert meta is mla_query_bmm.META
    assert set(mla_query_bmm.__all__) == set(meta.entry_points) | {"META"}
    for name in meta.entry_points:
        assert getattr(mla_query_bmm, name) is getattr(api, name)


def test_query_bmm_clear_caches_drops_compiled_launches(monkeypatch):
    launches = {0: object()}
    monkeypatch.setattr(api, "_LAUNCHES", launches)
    mla_query_bmm.clear_caches()
    assert launches == {}
