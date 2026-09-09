"""Opaque attention calls preserve ordering and reject released channel handles."""

from types import SimpleNamespace

import pytest
import torch

from b12x.comm.pcie import pcie_dcp_attention as module


def test_compiled_attention_calls_keep_channel_identity_and_lifetime():
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

    def run(query, gathered, lse, out):
        channel.query(query, gathered)
        channel.combine(gathered, lse, out)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    compiled(query, gathered, lse, out)
    assert calls == [("query", "target-attention"), ("combine", "target-attention")]
    torch.testing.assert_close(out, query)
    channel.close()
    with pytest.raises(RuntimeError, match="closed or released"):
        compiled(query, gathered, lse, out)
