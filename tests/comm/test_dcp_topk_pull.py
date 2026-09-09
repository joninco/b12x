"""Peer candidate selection preserves exact sets and deterministic graph output."""

import pytest
import torch

from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)
from b12x.comm.pcie._dcp_topk_pull_cute import (
    precompile_peer_topk,
    select_peer_topk,
)
from benchmarks.dcp_transport.fixtures import rank_inputs, references


@pytest.mark.parametrize("topk", [512, 2048])
def test_peer_selector_exact_repeatable_all_live_rows(topk):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    compiled = precompile_peer_topk(topk, 4, device.index)
    slabs = torch.empty((4, 16, topk + 8, 2), device=device)
    pointers = tuple(slabs[r].data_ptr() for r in range(4))
    output = torch.full((16, topk + 8), -7, dtype=torch.int32, device=device)
    freeze_kernel_resolution(
        "Peer top-k reuses static geometry across rows 1 through 16"
    )
    try:
        for rows in range(1, 17):
            assert precompile_peer_topk(topk, 4, device.index) is compiled
            out = output[:rows, :topk]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                select_peer_topk(pointers, out, slabs.stride(1))
            address = out.data_ptr()
            for seed in (951, 1953):
                ranks = [rank_inputs(r, rows, topk=topk, seed=seed) for r in range(4)]
                expected = references(ranks, 0)[2].sort(dim=1).values
                for rank, values in enumerate(ranks):
                    slabs[rank, :rows, :topk].copy_(values.packed)
                graph.replay()
                torch.cuda.synchronize()
                baseline = out.clone()
                torch.testing.assert_close(out.cpu().sort(dim=1).values, expected)
                allocated = torch.cuda.memory_allocated()
                for _ in range(30):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out, baseline)
                    assert out.data_ptr() == address
                    assert torch.cuda.memory_allocated() == allocated
                assert torch.all(output[:, topk:] == -7)
    finally:
        unfreeze_kernel_resolution()
