"""Candidate benchmark consumers select exact sets from both storage layouts."""

import pytest
import torch

from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)
from benchmarks.dcp_transport.fixtures import rank_inputs, references
from benchmarks.dcp_transport.selectors import precompile, select_owner, select_peers


@pytest.mark.parametrize("splits", [8, 32])
def test_sparse_attention_peer_queries_match_local_queries(splits):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference
    from benchmarks.dcp_transport.query_consumer import consume
    from benchmarks.dcp_transport.query_consumer import precompile as prepare_consumer

    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator().manual_seed(672)
    topk = 2048
    kv = pack_mla_kv_cache_reference(
        torch.randn((topk, 512), generator=generator).to(device),
        torch.randn((topk, 64), generator=generator).bfloat16().to(device),
    ).flatten()
    peers = [
        torch.empty((16, 8, 576), dtype=torch.bfloat16, device=device) for _ in range(4)
    ]
    pointers = tuple(p.data_ptr() for p in peers)
    local_query = torch.empty((16, 32, 576), dtype=torch.bfloat16, device=device)
    poison_query = torch.full_like(local_query, torch.nan)
    selection = torch.arange(topk, dtype=torch.int32, device=device).repeat(16, 1)
    selection[:, 1::4] = -1
    selection[0] = -1
    out = torch.empty((16, 32, splits, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((16, 32, splits), dtype=torch.float32, device=device)
    expected_out, expected_lse = torch.empty_like(out), torch.empty_like(lse)
    for peer_reads in (False, True):
        prepare_consumer(
            splits=splits, peer_reads=peer_reads, device_index=device.index
        )
    freeze_kernel_resolution("Sparse attention peer-query live row reuse")
    try:
        for rows in (1, 2, 4, 8, 16):
            for peer in peers:
                peer.copy_(torch.randn(peer.shape, generator=generator).bfloat16())
            local_query.copy_(torch.cat(peers, dim=1))
            consume(
                local_query[:rows],
                (),
                kv,
                selection[:rows],
                expected_out[:rows],
                expected_lse[:rows],
            )
            consume(
                poison_query[:rows],
                pointers,
                kv,
                selection[:rows],
                out[:rows],
                lse[:rows],
            )
            torch.testing.assert_close(out[:rows], expected_out[:rows], rtol=0, atol=0)
            torch.testing.assert_close(lse[:rows], expected_lse[:rows], rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                consume(
                    poison_query[:rows],
                    pointers,
                    kv,
                    selection[:rows],
                    out[:rows],
                    lse[:rows],
                )
            # Mutate each source independently after capture. A stale gathered
            # query or incorrect head-to-rank mapping cannot satisfy this oracle.
            for rank, peer in enumerate(peers):
                peer.add_(0.125 * (rank + 1))
            local_query.copy_(torch.cat(peers, dim=1))
            consume(
                local_query[:rows],
                (),
                kv,
                selection[:rows],
                expected_out[:rows],
                expected_lse[:rows],
            )
            allocated = torch.cuda.memory_allocated()
            for _ in range(30):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(out[:rows], expected_out[:rows], rtol=0, atol=0)
            torch.testing.assert_close(lse[:rows], expected_lse[:rows], rtol=0, atol=0)
    finally:
        unfreeze_kernel_resolution()


def test_peer_lse_merge_matches_float64_with_empty_shards():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from cutlass import BFloat16, Float32
    from b12x._lib.utils import current_cuda_stream
    from benchmarks.dcp_transport.publication import _merger, _ptr, _publisher

    device = torch.device("cuda", torch.cuda.current_device())
    for rank in range(4):
        _merger(rank, device.index)
        for shape in ((2304, 0), (4096, 0), (2304, 4096), (8192, 32)):
            _publisher(rank, *shape, device.index)
    freeze_kernel_resolution("Peer LSE merge live row counts")
    try:
        for rows in (1, 2, 4, 8, 16):
            ranks = [rank_inputs(rank, rows) for rank in range(4)]
            partials = [r.partial.to(device) for r in ranks]
            lses = [r.lse.to(device) for r in ranks]
            pointers = tuple(_ptr(BFloat16, p.data_ptr()) for p in partials)
            lse_pointers = tuple(_ptr(Float32, p.data_ptr()) for p in lses)
            out = torch.empty((rows, 8, 512), dtype=torch.bfloat16, device=device)
            for rank in range(4):
                fn = _merger(rank, device.index)
                expected = references(ranks, rank)[1]
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    fn(
                        pointers,
                        lse_pointers,
                        _ptr(BFloat16, out.data_ptr()),
                        rows,
                        current_cuda_stream(),
                    )
                allocated = torch.cuda.memory_allocated()
                for _ in range(30):
                    graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_allocated() == allocated
                torch.testing.assert_close(out.cpu(), expected, rtol=0.02, atol=0.02)
                assert torch.isfinite(out).all()
    finally:
        unfreeze_kernel_resolution()


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
