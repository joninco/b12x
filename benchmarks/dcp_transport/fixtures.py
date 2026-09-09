"""CPU-built correctness oracles for DCP transport benchmarks."""

from dataclasses import dataclass

import torch


@dataclass
class Inputs:
    query: torch.Tensor
    partial: torch.Tensor
    lse: torch.Tensor
    local_ids: torch.Tensor
    scores: torch.Tensor
    packed: torch.Tensor


def rank_inputs(
    rank: int, rows: int, *, seed: int = 731, topk: int = 2048, world_size: int = 4
) -> Inputs:
    """Generate disjoint rank data, score ties and empty partial shards."""
    generator = torch.Generator().manual_seed(seed + rank * 1009)
    query = torch.randn((rows, 8, 576), generator=generator).bfloat16()
    partial = torch.randn((rows, world_size * 8, 512), generator=generator).bfloat16()
    lse = torch.randn((rows, world_size * 8), generator=generator) * 20
    # Every rank but rank zero is empty for the first row. Poison verifies that
    # a zero-weight partial cannot contaminate a valid peer's result.
    if rank:
        lse[0] = -torch.inf
        partial[0] = torch.nan
    if rows > 1:
        lse[1] = -torch.inf
        partial[1] = torch.nan
    local_ids = torch.arange(topk, dtype=torch.int32).repeat(rows, 1)
    scores = torch.randint(-8, 9, (rows, topk), generator=generator).float()
    local_ids[0, topk // 8 :] = -1
    if rows > 1:
        local_ids[1] = -1
    scores[local_ids < 0] = -torch.inf
    global_ids = torch.where(local_ids >= 0, local_ids * world_size + rank, -1)
    return Inputs(
        query,
        partial,
        lse,
        local_ids,
        scores,
        torch.stack((scores, global_ids.float()), -1),
    )


def references(ranks: list[Inputs], rank: int):
    """Return exact query/set oracles and an independent float64 LSE merge."""
    query = torch.cat([r.query for r in ranks], dim=1)
    partial = torch.stack([r.partial for r in ranks]).double()
    lse = torch.stack([r.lse for r in ranks]).double()
    nonempty = torch.isfinite(lse)
    maximum = lse.max(dim=0).values
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = torch.where(nonempty, torch.exp(lse - maximum), 0)
    denominator = weights.sum(dim=0)
    partial = torch.where(nonempty[..., None], partial, 0)
    combined = (partial * weights[..., None]).sum(dim=0)
    combined /= denominator.clamp_min(torch.finfo(torch.float64).tiny)[..., None]
    combined = combined[:, rank * 8 : (rank + 1) * 8].bfloat16()
    candidates = torch.cat([r.packed for r in ranks], dim=1)
    rows, topk = ranks[0].scores.shape
    selected = torch.full((rows, topk), -1, dtype=torch.int32)
    for row in range(rows):
        valid = candidates[row, :, 1] >= 0
        values = candidates[row, valid]
        values = values[torch.argsort(values[:, 1], stable=True)]
        order = torch.argsort(values[:, 0], descending=True, stable=True)[:topk]
        selected[row, : len(order)] = values[order, 1].int()
    return query, combined, selected.sort(dim=-1).values
