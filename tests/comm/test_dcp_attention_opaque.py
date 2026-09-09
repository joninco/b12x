"""Opaque attention calls preserve ordering and reject released channel handles."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import pcie_dcp_attention as module


@pytest.mark.parametrize("masked", [False, True])
def test_compiled_attention_calls_keep_channel_identity_and_lifetime(
    monkeypatch, masked
):
    channel = object.__new__(module.PCIeDCPAttention)
    channel.device = torch.device("cpu")
    channel.channel_id = "target-attention"
    channel._closed = False
    channel._state = torch.zeros(4, dtype=torch.uint8)
    channel._handle = next(module._HANDLES)
    module._CHANNELS[channel._handle] = channel
    calls = []

    def gather(query, out, *, channel_id):
        calls.append(("query", channel_id))
        out.copy_(query.repeat(1, 4, 1))

    def combine(partial, lse, out, *, channel_id, is_lse_base_on_e):
        assert is_lse_base_on_e
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
    from b12x._lib.runtime_control import (
        freeze_kernel_resolution,
        unfreeze_kernel_resolution,
    )
    from b12x.comm.pcie._dcp_attention_metadata import (
        mask_local_lse,
        precompile_local_lse_mask,
    )

    device = torch.device("cuda", torch.cuda.current_device())
    lse = torch.randn((16, 64), device=device)[:, ::2]
    lengths = torch.empty(16, dtype=torch.int32, device=device)
    output = torch.empty((16, 32), device=device)
    kernel = precompile_local_lse_mask(32, device.index)
    freeze_kernel_resolution("Per-query LSE mask reuses one static head geometry")
    try:
        for rows in range(1, 17):
            assert precompile_local_lse_mask(32, device.index) is kernel
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                mask_local_lse(lse[:rows], lengths[:rows], output[:rows])
            for version in range(2):
                lengths.copy_((torch.arange(16, device=device) + version) % 4)
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
    finally:
        unfreeze_kernel_resolution()
