"""Opaque attention calls preserve ordering and reject released channel handles."""

from types import SimpleNamespace
from contextlib import contextmanager

import pytest
from b12x._lib.runtime_control import kernel_resolution_guard

import torch

from b12x.comm.pcie import pcie_dcp_attention as module
from b12x.comm.pcie.pcie_dcp_a2a import PCIeDCPA2A


@pytest.fixture
def prepared_transport(monkeypatch):
    observed = {}

    def prepare(runtime, calls, *, channel_id, ranks):
        observed.update(calls)
        assert tuple(ranks) == (0, 1, 2, 3)
        return SimpleNamespace(close=lambda: None), {name: object() for name in calls}

    monkeypatch.setattr(module, "_prepare_transport_calls", prepare)
    monkeypatch.setattr(
        module.dist, "get_process_group_ranks", lambda group: list(range(4))
    )
    return observed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("interleave", [1, 64])
@pytest.mark.parametrize("inplace", [False, True])
def test_local_lengths_reuse_compiled_geometry_and_replay(rank, interleave, inplace):
    from b12x.comm.pcie._dcp_attention_metadata import (
        localize_dcp_sequence_lengths,
        precompile_dcp_sequence_lengths,
    )

    device = torch.cuda.current_device()
    compiled = precompile_dcp_sequence_lengths(device)
    with kernel_resolution_guard("DCP causal lengths must not specialize live rows"):
        for rows in (1, 2, 3, 4, 8, 16, 127, 128, 129):
            values = torch.arange(rows, dtype=torch.int32) * 67
            seed = values.to(device)
            source = torch.empty_like(seed)
            out = source if inplace else torch.empty_like(seed)

            def run():
                source.copy_(seed)
                localize_dcp_sequence_lengths(source, out, 4, rank, interleave)

            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            values += 1
            seed.copy_(values)
            cycle = 4 * interleave
            expected = values // cycle * interleave + (
                values % cycle - rank * interleave
            ).clamp(0, interleave)
            pointer = out.data_ptr()
            for _ in range(3):
                graph.replay()
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                assert out.data_ptr() == pointer
                assert precompile_dcp_sequence_lengths(device) is compiled


def test_model_load_preparation_leaves_stream_ownership_for_graph_capture(
    monkeypatch, prepared_transport
):
    class Runtime:
        _stream_affine = True
        _owner_stream_key = None
        rank = 0
        _signal_ptrs = [100]
        _staging1_ptrs = [200]
        _slot_bytes = 100
        _bind_stream_key = PCIeDCPA2A._bind_stream_key

    runtime = Runtime()

    class Pool:
        _logical_channels = {"target": runtime}

        def prepare_channels(self, ids):
            assert ids == ("target",)

        def for_stream(self, **kwargs):
            runtime._bind_stream_key(1)
            return runtime

        @contextmanager
        def capture(self, **kwargs):
            runtime._bind_stream_key(2)
            yield

        def close(self):
            pass

    monkeypatch.setattr(
        module.PCIeDCPA2APool, "from_process_group", lambda **kw: Pool()
    )
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(module, "_normalize_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(module, "precompile_local_lse_mask", lambda *args: None)
    monkeypatch.setattr(
        module,
        "_tensor_from_cuda_pointer",
        lambda *a, **kw: torch.zeros(4, dtype=torch.uint8),
    )
    channel = module.PCIeDCPAttention(
        process_group=object(), device="cpu", channel_id="target"
    )
    assert runtime._owner_stream_key is None
    with channel.capture():
        assert runtime._owner_stream_key == 2
    channel.close()


@pytest.mark.parametrize("max_rows", [1, 16, 24, 32, 64])
def test_channel_capacity_sizes_the_pool_and_rejects_no_positive_value(
    monkeypatch, max_rows, prepared_transport
):
    """Any positive capacity reaches the pool as its batch size."""
    requested = {}

    class Runtime:
        rank = 0
        _signal_ptrs = [100]
        _staging1_ptrs = [200]
        _slot_bytes = 100

    class Pool:
        _logical_channels = {"target": Runtime()}

        def prepare_channels(self, ids):
            pass

        def close(self):
            pass

    def from_process_group(**kwargs):
        requested.update(kwargs)
        return Pool()

    monkeypatch.setattr(module.PCIeDCPA2APool, "from_process_group", from_process_group)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(module, "_normalize_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(module, "precompile_local_lse_mask", lambda *args: None)
    monkeypatch.setattr(
        module,
        "_tensor_from_cuda_pointer",
        lambda *a, **kw: torch.zeros(4, dtype=torch.uint8),
    )
    channel = module.PCIeDCPAttention(
        process_group=object(), device="cpu", channel_id="target", max_rows=max_rows
    )
    assert requested["max_batch_size"] == max_rows
    assert channel.max_rows == max_rows
    assert (channel.threads, channel.block_limit) == (512, 16)
    assert prepared_transport["all_gather_heads"]["threads"] == 512
    assert prepared_transport["lse_reduce_scatter"]["threads"] == 512
    channel.close()
    with pytest.raises(ValueError, match="positive row capacity"):
        module.PCIeDCPAttention(
            process_group=object(), device="cpu", channel_id="target", max_rows=0
        )


@pytest.mark.parametrize("masked", [False, True])
def test_compiled_attention_calls_keep_channel_identity_and_lifetime(
    monkeypatch, masked
):
    channel = object.__new__(module.PCIeDCPAttention)
    channel.device = torch.device("cpu")
    channel.channel_id = "target-attention"
    channel.threads, channel.block_limit = 512, 16
    channel._closed = False
    channel._session = None
    channel._plans = {"all_gather_heads": object(), "lse_reduce_scatter": object()}
    channel._state = torch.zeros(4, dtype=torch.uint8)
    channel._handle = next(module._HANDLES)
    module._CHANNELS[channel._handle] = channel
    calls = []

    def gather(query, out, *, plan, channel_id, threads, block_limit):
        assert plan is channel._plans["all_gather_heads"]
        assert (threads, block_limit) == (512, 16)
        calls.append(("query", channel_id))
        out.copy_(query.repeat(1, 4, 1))

    def combine(
        partial, lse, out, *, plan, channel_id, is_lse_base_on_e, threads, block_limit
    ):
        assert plan is channel._plans["lse_reduce_scatter"]
        assert is_lse_base_on_e
        assert (threads, block_limit) == (512, 16)
        calls.append(("combine", channel_id))
        out.copy_(partial[:, :1])

    channel._pool = SimpleNamespace(
        all_gather_heads=gather, lse_reduce_scatter=combine, close=lambda: None
    )
    query = torch.ones((1, 1, 8), dtype=torch.bfloat16)
    gathered = torch.empty((1, 4, 8), dtype=torch.bfloat16)
    lse = torch.zeros((1, 4), dtype=torch.float32)
    out = torch.empty_like(query)
    scratch = torch.empty_like(lse)
    lengths = torch.ones(1, dtype=torch.int32)

    def mask(lse, local_lengths, masked_lse):
        assert local_lengths is lengths
        masked_lse.copy_(lse)

    monkeypatch.setattr(module, "mask_local_lse", mask)

    def run(query, gathered, lse, out):
        channel.query(query, gathered)
        if masked:
            channel.combine_masked(gathered, lse, lengths, scratch, out)
        else:
            channel.combine(gathered, lse, out)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    compiled(query, gathered, lse, out)
    assert calls == [("query", "target-attention"), ("combine", "target-attention")]
    torch.testing.assert_close(out, query)
    channel.close()
    with pytest.raises(RuntimeError, match="closed or released"):
        compiled(query, gathered, lse, out)


def test_local_lse_mask_uses_each_query_length_under_frozen_graph_replay():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.comm.pcie._dcp_attention_metadata import (
        mask_local_lse,
        precompile_local_lse_mask,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    capacity = 64
    lse = torch.randn((capacity, 64), device=device)[:, ::2]
    lengths = torch.empty(capacity, dtype=torch.int32, device=device)
    output = torch.empty((capacity, 32), device=device)
    kernel = precompile_local_lse_mask(32, device.index)
    with kernel_resolution_guard("Per-query LSE mask reuses one static head geometry"):
        # Rows 1 through 16 plus the larger uniform decode graph sizes up to
        # the 64-row transport capacity.
        for rows in (*range(1, 17), 24, 32, 40, 48, 56, 64):
            assert precompile_local_lse_mask(32, device.index) is kernel
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mask_local_lse(lse[:rows], lengths[:rows], output[:rows])
            for version in range(2):
                lengths.copy_((torch.arange(capacity, device=device) + version) % 4)
                lse.normal_()
                lse[lengths == 0] = float("nan")
                expected = lse.masked_fill(lengths[:, None] == 0, -torch.inf)
                allocated = torch.cuda.memory_allocated()
                for _ in range(30):
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                torch.testing.assert_close(
                    output[:rows], expected[:rows], rtol=0, atol=0
                )


@pytest.mark.parametrize("fail", (False, True))
def test_transport_preparation_restores_affinity_without_claiming_a_stream(
    monkeypatch, fail
):
    from b12x import preparation
    from b12x.comm.pcie._dcp_preparation import _prepare_transport_calls

    class Runtime:
        rank, world_size = 0, 4
        device = torch.device("cpu")
        max_batch_size, total_heads, head_dim, query_head_dim = 64, 32, 512, 576
        _slot_bytes, _signal_ptrs = 4096, (100, 200, 300, 400)
        _owner_stream_key, _stream_affine = None, True
        _bind_stream_key = PCIeDCPA2A._bind_stream_key

        def _resolve_launch_config(self, *, threads, block_limit):
            return threads, block_limit

    runtime = Runtime()
    events = []

    class Session:
        def __init__(self, **kwargs):
            assert kwargs["compile_workers"] == 0

        def prepare(self, requests, *, coordinator):
            runtime._bind_stream_key(17)
            assert runtime._owner_stream_key is None
            (request,) = requests
            assert request.plan.query.setup["max_batch_size"] == 64
            assert request.plan.query.call["threads"] == 512
            assert (
                coordinator(SimpleNamespace(ready_collectives=(request.collective,)))
                == request.collective.key
            )
            if fail:
                raise RuntimeError("priming failed")
            events.append("primed")

        def close(self):
            events.append("closed")

    monkeypatch.setattr(preparation, "PreparationSession", Session)
    calls = {
        "all_gather_heads": {
            "local_input": torch.empty((64, 8, 576), dtype=torch.bfloat16),
            "out": torch.empty((64, 32, 576), dtype=torch.bfloat16),
            "threads": 512,
            "block_limit": 16,
        }
    }
    if fail:
        with pytest.raises(RuntimeError, match="priming failed"):
            _prepare_transport_calls(
                runtime, calls, channel_id="target", ranks=range(4)
            )
        assert events == ["closed"]
    else:
        session, plans = _prepare_transport_calls(
            runtime, calls, channel_id="target", ranks=range(4)
        )
        assert set(plans) == {"all_gather_heads"}
        assert events == ["primed"]
        session.close()
    assert runtime._stream_affine
    assert runtime._owner_stream_key is None
    runtime._bind_stream_key(29)
    assert runtime._owner_stream_key == 29
