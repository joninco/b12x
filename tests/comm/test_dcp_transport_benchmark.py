"""Candidate benchmark consumers select exact sets from both storage layouts."""

import pytest
import torch

from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)
from benchmarks.dcp_transport.fixtures import rank_inputs, references
from benchmarks.dcp_transport.selectors import precompile, select_owner, select_peers


@pytest.mark.parametrize("layout", ["owner_planes", "peer_pairs"])
@pytest.mark.parametrize("topk", [512, 2048])
def test_transport_candidate_consumers_reuse_static_compile(layout, topk):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    fn = precompile(topk, 4, layout, device.index)
    freeze_kernel_resolution("DCP benchmark consumers across live row counts")
    try:
        for rows in (1, 2, 4, 8, 16):
            ranks = [rank_inputs(rank, rows, topk=topk) for rank in range(4)]
            expected = references(ranks, 0)[2]
            output_storage = torch.empty(
                (rows, topk + 8), device=device, dtype=torch.int32
            )
            out = output_storage[:, :topk]
            if layout == "owner_planes":
                # Padded row strides ensure the consumer does not assume that
                # planes occupy exactly the live candidate extent.
                planes = torch.empty((2, rows, 4 * topk + 16), device=device)
                scores = planes[0, :, : 4 * topk]
                indices = planes[1].view(torch.int32)[:, : 4 * topk]
                scores.copy_(torch.cat([r.scores for r in ranks], dim=1))
                indices.copy_(torch.cat([r.packed[..., 1] for r in ranks], 1).int())

                def run():
                    select_owner(indices, scores, out, 4)
            else:
                slabs = torch.empty((4, rows, topk + 8, 2), device=device)
                for rank, values in enumerate(ranks):
                    slabs[rank, :, :topk].copy_(values.packed)
                pointers = tuple(slabs[r].data_ptr() for r in range(4))

                def run():
                    select_peers(pointers, out, slabs.stride(1))

            run()
            torch.testing.assert_close(out.cpu().sort().values, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            before = torch.cuda.memory_allocated()
            for _ in range(30):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == before
            torch.testing.assert_close(out.cpu().sort().values, expected)
            assert precompile(topk, 4, layout, device.index) is fn
    finally:
        unfreeze_kernel_resolution()
