"""Rank-major candidate selection: exact sets, strided storage and graph reuse."""

import pytest
import torch

from b12x._lib.runtime_control import (
    freeze_kernel_resolution,
    unfreeze_kernel_resolution,
)
from b12x.comm.pcie.dcp_candidate_topk import (
    pack_dcp_candidates,
    precompile_rank_major_topk,
    rank_major_topk,
)


@pytest.mark.parametrize("topk", [512, 1024, 2048])
@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_rank_major_selection_matches_reference_with_frozen_resolution(
    topk, world_size
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda", torch.cuda.current_device())
    precompile_rank_major_topk(topk, world_size, device)
    generator = torch.Generator().manual_seed(731)
    storage = torch.empty((world_size, 19, topk, 2), device=device)
    output_storage = torch.empty((16, topk + 8), dtype=torch.int32, device=device)
    freeze_kernel_resolution("rank-major top-k row-count reuse")
    try:
        for rows in (1, 2, 4, 8, 16):
            # Slice keeps a padded rank stride and a padded output row stride.
            gathered = storage[:, :rows]
            out = output_storage[:rows, :topk]
            scores = torch.randint(
                -8, 9, (rows, world_size * topk), generator=generator
            )
            ids = torch.arange(world_size * topk).expand(rows, -1).clone()
            ids[0, topk // 2 :] = -1  # Underfull row with empty peer shards.
            if rows > 1:
                ids[1] = -1  # Entirely empty row.
            expected = torch.full((rows, topk), -1, dtype=torch.int32)
            for row in range(rows):
                valid = ids[row] >= 0
                order = torch.argsort(scores[row, valid], descending=True, stable=True)
                selected = ids[row, valid][order[:topk]]
                expected[row, : len(selected)] = selected
            packed = torch.stack((scores.float(), ids.float()), dim=-1)
            gathered.copy_(packed.reshape(rows, world_size, topk, 2).transpose(0, 1))
            rank_major_topk(gathered, out)
            torch.testing.assert_close(out.cpu().sort().values, expected.sort().values)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                rank_major_topk(gathered, out)
            allocated = torch.cuda.memory_allocated()
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            torch.testing.assert_close(out.cpu().sort().values, expected.sort().values)
    finally:
        unfreeze_kernel_resolution()


def test_rank_major_selection_handles_prefill_chunk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, topk, ranks = 8192, 2048, 4
    gathered = torch.empty((ranks, rows, topk, 2), device="cuda")
    ids = torch.arange(topk, device="cuda")
    for rank in range(ranks):
        gathered[rank, :, :, 0] = rank
        gathered[rank, :, :, 1] = ids + rank * topk
    out = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    rank_major_topk(gathered, out)
    expected = (ids + (ranks - 1) * topk).to(torch.int32).expand(rows, -1)
    torch.testing.assert_close(out.sort().values, expected)


@pytest.mark.parametrize("interleave", [1, 4])
def test_packed_local_candidates_select_global_positions(interleave):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rows, topk, ranks = 4, 2048, 4
    local_ids = (
        torch.arange(topk, dtype=torch.int32, device="cuda").expand(rows, -1).clone()
    )
    local_ids[0, 2:] = -1
    local_ids[1] = -1
    scores = torch.ones((rows, topk), device="cuda")
    gathered = torch.empty((ranks, rows, topk, 2), device="cuda")
    for rank in range(ranks):
        pack_dcp_candidates(local_ids, scores, gathered[rank], rank, ranks, interleave)
        ids = local_ids.cpu().to(torch.int64)
        expected_ids = (
            ids // interleave * (ranks * interleave)
            + rank * interleave
            + ids % interleave
        )
        expected_ids[ids < 0] = -1
        expected_scores = torch.where(ids >= 0, 1.0, -torch.inf)
        torch.testing.assert_close(gathered[rank, :, :, 1].cpu(), expected_ids.float())
        torch.testing.assert_close(gathered[rank, :, :, 0].cpu(), expected_scores)
    output = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    rank_major_topk(gathered, output)
    valid_ids = gathered[:, :, :, 1].permute(1, 0, 2).reshape(rows, -1).cpu().int()
    # Every score ties; ascending global ids are the exact selected set.
    expected = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        selected = valid_ids[row][valid_ids[row] >= 0].sort().values[:topk]
        expected[row, : len(selected)] = selected
    torch.testing.assert_close(output.cpu().sort().values, expected.sort().values)
