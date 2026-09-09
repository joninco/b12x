"""Publication prototypes and matched sparse-attention consumers."""

from contextlib import nullcontext

import torch

from b12x.attention._shared.mla.reference import pack_mla_kv_cache_reference
from b12x.comm.pcie.dcp_candidate_topk import pack_dcp_candidates
from benchmarks.dcp_transport.cases import Case
from benchmarks.dcp_transport.fixtures import rank_inputs, references
from benchmarks.dcp_transport.publication import PublicationChannel
from benchmarks.dcp_transport.query_consumer import consume, precompile
from benchmarks.dcp_transport.selectors import precompile as prepare_selector
from benchmarks.dcp_transport.selectors import select_peers


def query_consumer_case(kind, group, device, rows, rank, *, splits=32):
    """Measure query access with identical sparse-attention arithmetic.

    The shared case consumes query and candidate publications independently,
    with the same fixed local sparse index list as the matched local consumer.
    It isolates exchange cost; it does not model indexer scoring, localization
    or the serving schedule between indexer selection and attention.
    """
    channel = None if kind == "local_consumer" else PublicationChannel(group, device)
    generator = torch.Generator().manual_seed(637 + rank)
    cache = pack_mla_kv_cache_reference(
        torch.randn((2048, 512), generator=generator).to(device),
        torch.randn((2048, 64), generator=generator).bfloat16().to(device),
    ).flatten()
    local = torch.empty((rows, 8, 576), dtype=torch.bfloat16, device=device)
    gathered = torch.empty((rows, 32, 576), dtype=torch.bfloat16, device=device)
    poison = torch.full_like(gathered, torch.nan)
    indices = torch.arange(2048, dtype=torch.int32, device=device).repeat(rows, 1)
    indices = torch.where(indices % 4 == rank, indices // 4, -1)
    output = torch.empty((rows, 32, splits, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((rows, 32, splits), dtype=torch.float32, device=device)
    reference_out, reference_lse = torch.empty_like(output), torch.empty_like(lse)
    expected_out, expected_lse = (
        torch.empty_like(output, device="cpu"),
        torch.empty_like(lse, device="cpu"),
    )
    ids = torch.empty((rows, 2048), dtype=torch.int32, device=device)
    scores = torch.empty((rows, 2048), dtype=torch.float32, device=device)
    packed = torch.empty((rows, 2048, 2), device=device)
    selected = torch.empty((rows, 2048), dtype=torch.int32, device=device)
    expected_selected = torch.empty_like(selected, device="cpu")
    for peer_reads in (False, True):
        precompile(splits=splits, peer_reads=peer_reads, device_index=device.index)
    prepare_selector(2048, 4, "peer_pairs", device.index)

    def refresh(seed):
        ranks = [rank_inputs(r, rows, seed=seed) for r in range(4)]
        local.copy_(ranks[rank].query)
        gathered.copy_(torch.cat([r.query for r in ranks], dim=1))
        ids.copy_(ranks[rank].local_ids)
        scores.copy_(ranks[rank].scores)
        expected_selected.copy_(references(ranks, rank)[2])
        consume(gathered, (), cache, indices, reference_out, reference_lse)
        expected_out.copy_(reference_out)
        expected_lse.copy_(reference_lse)

    refresh(731)

    def run():
        if kind == "local_consumer":
            consume(gathered, (), cache, indices, output, lse)
        else:
            if kind == "shared_consumers":
                pack_dcp_candidates(ids, scores, packed, rank, 4, 1)
                channel.publish("shared", local, packed)
                select_peers(channel.candidate_pointers, selected, 4096)
            else:
                channel.publish("query", local)
            consume(poison, channel.query_pointers, cache, indices, output, lse)

    outputs, expected = (output, lse), (expected_out, expected_lse)
    if kind == "shared_consumers":
        outputs, expected = (*outputs, selected), (*expected, expected_selected)
    return Case(
        kind,
        "consumers",
        run,
        outputs,
        expected,
        channel.capture if channel else nullcontext,
        [channel] if channel else [],
        {
            "sparse_heads": 32,
            "query_dim": 576,
            "topk": 2048,
            "valid_candidates": 512,
            "splits": splits,
            "planned_rows": 16,
            "local_consumer_subtraction_required": kind != "local_consumer",
            "shared_consumer_scope": "Independent attention and candidate consumers share publication; localization is outside the benchmark.",
        },
        refresh,
    )


def publication_combine_case(group, device, rows, rank):
    channel = PublicationChannel(group, device)
    partial = torch.empty((rows, 32, 512), dtype=torch.bfloat16, device=device)
    lse = torch.empty((rows, 32), dtype=torch.float32, device=device)
    output = torch.empty((rows, 8, 512), dtype=torch.bfloat16, device=device)
    expected = torch.empty_like(output, device="cpu")

    def refresh(seed):
        ranks = [rank_inputs(r, rows, seed=seed) for r in range(4)]
        partial.copy_(ranks[rank].partial)
        lse.copy_(ranks[rank].lse)
        expected.copy_(references(ranks, rank)[1])

    def run():
        channel.publish("partial", partial, lse)
        channel.combine(output)

    refresh(731)
    return Case(
        "published_combine",
        "combine",
        run,
        (output,),
        (expected,),
        channel.capture,
        [channel],
        {
            "payload_transport": "local publication and plain peer loads",
            "publication_blocks": 16,
        },
        refresh,
    )


def publication_candidate_case(group, device, rows, rank):
    """Isolate candidate publication for comparison with shared publication."""
    channel = PublicationChannel(group, device)
    ids = torch.empty((rows, 2048), dtype=torch.int32, device=device)
    scores = torch.empty((rows, 2048), dtype=torch.float32, device=device)
    packed = torch.empty((rows, 2048, 2), device=device)
    selected = torch.empty_like(ids)
    expected = torch.empty_like(ids, device="cpu")
    prepare_selector(2048, 4, "peer_pairs", device.index)

    def refresh(seed):
        ranks = [rank_inputs(r, rows, seed=seed) for r in range(4)]
        ids.copy_(ranks[rank].local_ids)
        scores.copy_(ranks[rank].scores)
        expected.copy_(references(ranks, rank)[2])

    def run():
        pack_dcp_candidates(ids, scores, packed, rank, 4, 1)
        channel.publish("candidates", packed)
        select_peers(channel.candidate_pointers, selected, 4096)

    refresh(731)
    return Case(
        "published_candidates",
        "candidates",
        run,
        (selected,),
        (expected,),
        channel.capture,
        [channel],
        {"publication_blocks": 16},
        refresh,
    )
