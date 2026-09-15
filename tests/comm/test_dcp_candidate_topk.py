"""Rank-major candidate selection: exact sets, strided storage and graph reuse."""

import math

import pytest
import torch

from b12x._lib.runtime_control import (
    kernel_resolution_guard,
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
    with kernel_resolution_guard("rank-major top-k row-count reuse"):
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


def _selected_id_reference(candidates: torch.Tensor, topk: int) -> torch.Tensor:
    """CPU set oracle with the selector's positive-zero-before-negative-zero rule."""
    ranks, rows, width, _ = candidates.shape
    result = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        valid = []
        seen = set()
        for score, token in candidates[:, row].reshape(ranks * width, 2).tolist():
            if token < 0:
                continue
            if math.isnan(score) or not token.is_integer() or token in seen:
                raise ValueError(
                    "Reference requires non-NaN scores and unique integer IDs"
                )
            seen.add(token)
            # Numeric comparison handles all finite values and infinities.
            # The bit-key selector distinguishes the two IEEE zero encodings.
            positive_zero = score == 0 and math.copysign(1.0, score) > 0
            valid.append((score, positive_zero, -int(token)))
        valid.sort(reverse=True)
        for index, (_, _, negative_id) in enumerate(valid[:topk]):
            result[row, index] = -negative_id
    return result


def test_selection_reference_distinguishes_signed_zero_and_global_id_ties():
    candidates = torch.tensor(
        [
            [[[0.0, 9], [-0.0, 1], [2.0, 8], [float("nan"), -1]]],
            [[[0.0, 3], [-0.0, 0], [2.0, 2], [-float("inf"), 7]]],
        ],
        dtype=torch.float32,
    )
    assert _selected_id_reference(candidates, 7).tolist() == [[2, 8, 3, 9, 0, 1, 7]]


@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_owner_row_views_preserve_exact_candidate_sets(topk):
    """Owner slices retain rank pitch while selecting the same changed candidates."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    ranks, rows, owner_rows = 4, 12, 3
    storage = torch.empty((ranks, rows + 5, topk, 2), device="cuda")
    full_output = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    owner_output = torch.empty((owner_rows, topk), dtype=torch.int32, device="cuda")
    generator = torch.Generator().manual_seed(941)
    for generation in range(2):
        scores = torch.randint(-2, 3, (ranks, rows, topk), generator=generator).float()
        scores[0, :, : topk // 2] = -0.0
        scores[1, :, : topk // 2] = 0.0
        ids = torch.randperm(ranks * topk, generator=generator).reshape(ranks, 1, topk)
        ids = ids.expand(-1, rows, -1).clone() + generation * ranks * topk
        ids[:, 0] = -1
        ids[1:, 1] = -1
        candidates = torch.stack((scores, ids.float()), dim=-1)
        expected = _selected_id_reference(candidates, topk)
        storage[:, :rows].copy_(candidates)
        rank_major_topk(storage[:, :rows], full_output)
        torch.testing.assert_close(
            full_output.cpu().sort().values, expected.sort().values
        )
        for owner in range(ranks):
            start = owner * owner_rows
            rank_major_topk(storage[:, start : start + owner_rows], owner_output)
            torch.testing.assert_close(
                owner_output.cpu().sort().values,
                expected[start : start + owner_rows].sort().values,
            )
