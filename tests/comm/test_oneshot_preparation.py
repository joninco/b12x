"""CPU contracts for preparing the 64-row fused collective and its launch ABI."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import _oneshot_cute, _oneshot_preparation as preparation
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReduce, _CuTeOneshotBackend
from tests.comm.test_pcie_oneshot import _make_cute_state


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.delenv("B12X_PCIE_FUSED_THREADS", raising=False)
    monkeypatch.delenv("B12X_PCIE_FUSED_CTAS_PER_ROW", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP8_OWNER_REDUCE", "0")
    monkeypatch.setattr(_CuTeOneshotBackend, "_device_index", staticmethod(lambda _: 0))
    backend = _CuTeOneshotBackend()
    native = _make_cute_state(8, eager_buffer_bytes=2 << 20)
    backend._states[7] = native
    monkeypatch.setattr(
        preparation, "_owned_resident_layout", lambda *_: (256, 4096, 1)
    )
    return SimpleNamespace(
        _ext=backend,
        _ptr=7,
        device=torch.device("cpu"),
        world_size=8,
        rank=0,
        max_size=2 << 20,
        fused_max_rows=64,
        rank_data_bytes=256,
        should_allreduce=lambda inp: inp.is_contiguous(),
    )


@pytest.mark.parametrize("rows", (16, 32, 64))
def test_fused_metadata_and_tensor_declarations_use_the_capacity_geometry(
    runtime, rows
):
    inp = torch.empty((rows, 6144), dtype=torch.bfloat16)
    surface = "OneshotAllReducePool.all_reduce_fused_add_rms_norm"
    query = preparation.query_from_metadata(
        runtime, surface=surface, shape=inp.shape, dtype=inp.dtype
    )
    actual = preparation.query_from_runtime(runtime, surface=surface, call={"inp": inp})
    assert actual == query
    assert (
        query.call["threads"],
        query.call["reg_packs"],
        query.call["ctas_per_row"],
    ) == (768, 1, 1)
    assert query.call["blocks"] == rows
    assert query.call["mode"] == "stage_scatter_gather_packed"


@pytest.mark.parametrize("rows", (0, 65, 80))
def test_fused_declaration_rejects_rows_outside_channel_capacity(runtime, rows):
    with pytest.raises(ValueError, match="supports 1 to 64 rows"):
        preparation.query_from_metadata(
            runtime,
            surface="OneshotAllReduce.all_reduce_fused_add_rms_norm",
            shape=(rows, 6144),
            dtype=torch.bfloat16,
        )


def test_oneshot_declaration_cannot_exceed_established_eager_capacity(runtime):
    runtime.max_size = 4096
    with pytest.raises(ValueError, match="established runtime capacity"):
        preparation.query_from_metadata(
            runtime,
            surface="OneshotAllReduce.all_reduce",
            shape=(1, 4096),
            dtype=torch.bfloat16,
        )


def test_fused_compiler_receives_frozen_register_geometry(runtime, monkeypatch):
    query = preparation.query_from_metadata(
        runtime,
        surface="OneshotAllReduce.all_reduce_fused_add_rms_norm",
        shape=(64, 6144),
        dtype=torch.bfloat16,
    )
    captured = []

    def compile_fused(
        dtype,
        world,
        rank,
        mode,
        single_cta,
        register_normalize,
        device_selection,
        slot,
        threads,
        reg_packs,
        device,
    ):
        captured.append((device_selection, slot, threads, reg_packs))
        return object()

    monkeypatch.setattr(_oneshot_cute, "get_fused_oneshot_launcher", compile_fused)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setenv("B12X_PCIE_FUSED_THREADS", "256")
    payload = {
        "surface": query.surface,
        "world_size": 8,
        "rank": 0,
        "topology": query.topology,
        "call": dict(query.call),
        "setup": dict(query.setup),
    }
    launchers = preparation.compile_oneshot_surface(payload, 0)
    assert set(launchers) == {(False, 0), (True, 0), (True, 1)}
    assert captured == [(False, 0, 768, 1), (True, 0, 768, 1), (True, 1, 768, 1)]


def test_prepared_fused_launch_passes_independent_residual_strides(
    runtime, monkeypatch
):
    inp = torch.empty((64, 6144), dtype=torch.bfloat16)
    residual = torch.empty((64, 12288), dtype=inp.dtype)[:, 6144:]
    residual_out = torch.empty((64, 18432), dtype=inp.dtype)[:, :6144]
    out = torch.empty_like(inp)
    weight = torch.ones(6144, dtype=inp.dtype)
    query = preparation.query_from_runtime(
        runtime,
        surface="OneshotAllReduce.all_reduce_fused_add_rms_norm",
        call={"inp": inp},
    )
    captured = []

    def launch(
        table,
        signals,
        src,
        res,
        w,
        dst,
        res_out,
        hidden,
        rows,
        ctas,
        res_stride,
        res_out_stride,
        shards,
        epsilon,
        blocks,
    ):
        captured.append((hidden, rows, ctas, res_stride, res_out_stride, blocks))

    native = runtime._ext._state(runtime._ptr)
    runtime._prepared_launcher = lambda *_: (launch, native)
    monkeypatch.setattr(runtime._ext, "_select_table", lambda *_: (123, True))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    PCIeOneshotAllReduce._run_prepared_fused(
        runtime,
        inp,
        residual,
        weight,
        out,
        residual_out,
        1e-6,
        query,
        {},
    )
    assert captured == [(768, 64, 1, 1536, 2304, 64)]


def test_fused_oneshot_capacity_variants_reuse_launchers_for_live_rows(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    oneshot = importlib.import_module("b12x.comm.pcie._oneshot_cute")
    pcie = importlib.import_module("b12x.comm.pcie.pcie_oneshot")
    runtime_control = importlib.import_module("b12x._lib.runtime_control")

    monkeypatch.delenv("B12X_PCIE_ONESHOT_PUSH", raising=False)
    monkeypatch.setenv("B12X_PCIE_TP8_OWNER_REDUCE", "0")
    state = pcie._CuTeOneshotState(
        rank=0,
        world_size=8,
        signal_ptrs=tuple(range(8)),
        rank_data=torch.empty(256, dtype=torch.uint8),
        signal_table_address=0,
        next_table_offset=128,
        registered_tables={},
        eager_tables=(300, 400),
        eager_buffer_bytes=256 * 1024,
        transport_policy=(False, False, False, False, False),
        scatter_gather_storage=True,
    )
    plans = [
        pcie._CuTeOneshotBackend._fused_launch_plan(
            state,
            torch.empty((rows, 6144), dtype=torch.bfloat16),
        )
        for rows in (3, 4, 5, 6, 8, 12, 16)
    ]

    compile_specs = []

    def compile_stub(*_args, **kwargs):
        compile_specs.append(kwargs["compile_spec"])
        from b12x._lib.compile_plan import attach_programs

        return attach_programs(lambda *_args: None)

    monkeypatch.setattr(oneshot, "b12x_compile", compile_stub)
    monkeypatch.setattr(oneshot, "current_cuda_stream", lambda: 0)
    monkeypatch.setattr(oneshot, "make_ptr", lambda *_args, **_kwargs: object())
    oneshot.get_fused_oneshot_launcher.cache_clear()
    oneshot._PREPARED_FUSED_ONESHOT_LAUNCHERS.clear()

    def resolve(plan):
        variant = plan.variant
        return oneshot.get_fused_oneshot_launcher(
            "bfloat16",
            8,
            0,
            variant.mode,
            variant.single_cta,
            variant.register_normalize,
            False,
            0,
            variant.threads,
            variant.reg_packs,
            0,
        )

    try:
        warmed = [resolve(plan) for plan in plans]
        assert len(compile_specs) == 2

        with runtime_control.kernel_resolution_guard(
            "fused one-shot live rows must reuse capacity launchers"
        ):
            reused = [resolve(plan) for plan in plans]

        assert all(
            actual is expected for actual, expected in zip(reused, warmed, strict=True)
        )
        assert len(compile_specs) == 2
    finally:
        oneshot.get_fused_oneshot_launcher.cache_clear()
        oneshot._PREPARED_FUSED_ONESHOT_LAUNCHERS.clear()
