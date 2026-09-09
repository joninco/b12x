#!/usr/bin/env python3
"""Correctness-gated local kernel measurements for GLM DCP4 prefill.

Native-copy and owner-selection cases isolate computation on one GPU. They do
not measure collective transport, prefetch overlap, or serving performance.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import pathlib
import statistics
import subprocess
import sys
from collections.abc import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from benchmarks.benchmark_mla_query_projection import balanced_samples_us

SOURCE_REFERENCE = {
    "b12x": "21b0a79b2ef9267afe7c96bf3f255f58980db700",
    "vllm": "310921207b777fb58df48f7c24037e55249e8cd4",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("query", "native-copy", "selection"), required=True
    )
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--runtime-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.rows <= 8192:
        parser.error("rows must be in 1..8192")
    if args.warmup < 1 or args.repetitions < 2:
        parser.error("warmup must be positive and repetitions at least two")
    if args.case in ("native-copy", "selection") and args.rows % 4:
        parser.error("native-copy and selection require rows divisible by four")
    if args.case == "native-copy" and (args.context < args.rows or args.context % 256):
        parser.error("native-copy context must cover rows and be divisible by 256")
    return args


def file_record(path):
    path = pathlib.Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git_record(root):
    command = ["git", "-C", str(root)]
    result = subprocess.run(
        command + ["rev-parse", "HEAD"], capture_output=True, text=True
    )
    if result.returncode:
        return {
            "available": False,
            "source_identity": "external runtime manifest and module hashes",
        }
    status = subprocess.check_output(command + ["status", "--porcelain"], text=True)
    if status:
        raise RuntimeError(
            f"Measurements require committed clean source: {root}\n{status}"
        )
    return {
        "available": True,
        "root": str(root),
        "head": result.stdout.strip(),
        "status": status,
    }


def query_case(args, device):
    from b12x.gemm import mla_query_bmm

    packed = torch.empty((args.rows, 8, 256), dtype=torch.bfloat16, device=device)
    query = packed[:, :, :192].permute(1, 0, 2)
    weight = torch.empty((8, 512, 192), dtype=torch.bfloat16, device=device).transpose(
        1, 2
    )
    output_storage = [
        torch.empty((args.rows, 8, 576), dtype=torch.bfloat16, device=device)
        for _ in range(2)
    ]
    baseline_out, candidate_out = [
        storage[:, :, :512].permute(1, 0, 2) for storage in output_storage
    ]

    def baseline():
        return torch.bmm(query, weight, out=baseline_out)

    def candidate():
        return mla_query_bmm.run(query, weight, candidate_out)

    for seed in (args.seed, args.seed + 1):
        torch.manual_seed(seed)
        query.normal_()
        weight.normal_()
        baseline()
        candidate()
        lhs, rhs = query.double(), weight.double()
        reference = torch.bmm(lhs, rhs)
        u = torch.finfo(torch.float32).eps / 2
        gamma = 192 * u / (1 - 192 * u)
        accumulation = gamma * torch.bmm(lhs.abs(), rhs.abs())
        bound = (
            accumulation
            + (reference.abs() + accumulation) * torch.finfo(torch.bfloat16).eps / 2
        )
        for output in (baseline_out, candidate_out):
            if not torch.all(
                (output.double() - reference).abs()
                <= bound + torch.finfo(torch.bfloat16).tiny
            ):
                raise AssertionError(
                    "Query projection exceeds FP32 accumulation plus BF16 rounding bound"
                )
    return (
        baseline,
        candidate,
        {
            "baseline": "torch_bmm_bf16",
            "candidate": "cute_query_bmm_bf16",
            "geometry": {"heads": 8, "rows": args.rows, "k": 192, "n": 512},
            "strides": {
                "query": query.stride(),
                "weight": weight.stride(),
                "output": candidate_out.stride(),
            },
            "correctness": "both changed-input seeds satisfy FP32 dot-product and BF16 rounding bound against FP64",
            "source_modules": [
                file_record(inspect.getfile(mla_query_bmm)),
                file_record(inspect.getfile(mla_query_bmm.run)),
                file_record(
                    pathlib.Path(inspect.getfile(mla_query_bmm.run)).with_name(
                        "_kernel.py"
                    )
                ),
            ],
        },
    )


def native_copy_case(args, device):
    from b12x.attention._shared.mla import kv_cache
    from vllm import _custom_ops as ops

    local = args.context // 4
    current = args.rows // 4
    page_size, record_bytes = 64, 656
    pages = local // page_size
    caches = [
        torch.empty((pages, page_size, record_bytes), dtype=torch.uint8, device=device)
        for _ in range(4)
    ]
    table = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
    rank_starts = torch.zeros((4, 1), dtype=torch.int32, device=device)
    rank_lengths = torch.full((4, 1), local, dtype=torch.int32, device=device)
    full = torch.tensor([args.context], dtype=torch.int32, device=device)
    queries = torch.tensor([0, args.rows], dtype=torch.int32, device=device)
    cumulative = torch.tensor([0, local], dtype=torch.int32, device=device)
    baseline_out = torch.empty(
        (4 * local, record_bytes), dtype=torch.uint8, device=device
    )
    candidate_out = torch.empty_like(baseline_out)
    compact = torch.empty((4 * current, record_bytes), dtype=torch.uint8, device=device)

    def baseline():
        for rank in range(4):
            ops.cp_gather_cache(
                src_cache=caches[rank],
                dst=baseline_out[rank * local : (rank + 1) * local],
                block_table=table,
                cu_seq_lens=cumulative,
                batch_size=1,
            )

    def candidate():
        for rank in range(4):
            kv_cache.gather_ckv_history(
                caches[rank],
                candidate_out[rank * local : (rank + 1) * local],
                table,
                rank_starts,
                rank_lengths,
                full,
                queries,
                dcp_rank=rank,
                dcp_world_size=4,
                interleave=1,
                num_reqs=1,
                padded_tokens=local,
            )
            kv_cache.gather_ckv_current_chunk(
                caches[rank],
                compact[rank * current : (rank + 1) * current],
                table,
                full,
                queries,
                dcp_rank=rank,
                dcp_world_size=4,
                interleave=1,
                num_reqs=1,
                current_capacity=current,
            )
        kv_cache.insert_ckv_current_chunk(
            compact,
            candidate_out,
            rank_starts,
            full,
            queries,
            dcp_world_size=4,
            interleave=1,
            num_reqs=1,
            current_capacity=current,
            padded_tokens=local,
        )

    for seed in (args.seed, args.seed + 1):
        torch.manual_seed(seed)
        for cache in caches:
            cache.random_(0, 256)
        baseline()
        candidate()
        expected = torch.cat([cache.view(-1, record_bytes) for cache in caches])
        if not torch.equal(baseline_out, expected) or not torch.equal(
            candidate_out, expected
        ):
            raise AssertionError(
                "Native history/chunk composition changed record bytes"
            )
    return (
        baseline,
        candidate,
        {
            "baseline": "four_rank_full_native_copy_on_one_gpu",
            "candidate": "four_rank_history_plus_compact_chunk_copy_on_one_gpu",
            "geometry": {
                "ranks": 4,
                "requests": 1,
                "context": args.context,
                "chunk_rows": args.rows,
                "record_bytes": 656,
                "interleave": 1,
            },
            "native_payload_bytes": args.context * record_bytes,
            "compact_chunk_payload_bytes": args.rows * record_bytes,
            "correctness": "both changed-byte seeds match an independent full-cache concatenation exactly",
            "limitations": "serialized local copies only; excludes DCP transport, asynchronous overlap, and producer computation",
            "source_modules": [
                file_record(inspect.getfile(kv_cache)),
                file_record(inspect.getfile(ops)),
            ],
        },
    )


def selection_case(args, device):
    from b12x.comm.pcie import dcp_candidate_topk as selection

    topk = 2048
    candidates = torch.empty(
        (4, args.rows, topk, 2), dtype=torch.float32, device=device
    )
    output = torch.empty((args.rows, topk), dtype=torch.int32, device=device)
    owner_rows = args.rows // 4
    owner_output = torch.empty((owner_rows, topk), dtype=torch.int32, device=device)
    ids = torch.arange(4 * topk, device=device).view(4, 1, topk)
    candidates[..., 1].copy_(ids)

    def baseline():
        return selection.rank_major_topk(candidates, output)

    def candidate():
        return selection.rank_major_topk(candidates[:, :owner_rows], owner_output)

    for seed in (args.seed, args.seed + 1):
        torch.manual_seed(seed)
        candidates[..., 0].random_(0, 17)
        baseline()
        candidate()
        scores = candidates[..., 0].permute(1, 0, 2).reshape(args.rows, -1)
        expected = torch.argsort(scores, dim=1, descending=True, stable=True)[
            :, :topk
        ].int()
        if not torch.equal(output.sort(dim=1).values, expected.sort(dim=1).values):
            raise AssertionError(
                "Replicated selection violated score/global-ID ordering"
            )
        if not torch.equal(
            owner_output.sort(dim=1).values, output[:owner_rows].sort(dim=1).values
        ):
            raise AssertionError(
                "Owner-row selected IDs differ from replicated selection"
            )
    return (
        baseline,
        candidate,
        {
            "baseline": "selection_over_all_replicated_rows",
            "candidate": "selection_over_one_owner_quarter_of_rows",
            "geometry": {
                "ranks": 4,
                "rows": args.rows,
                "owner_rows": owner_rows,
                "topk": topk,
            },
            "correctness": "two changed-score seeds with ties match stable score/global-ID selected sets",
            "limitations": "local selection work per rank only; excludes candidate exchange, restoration and aggregate rank elapsed time",
            "source_modules": [file_record(inspect.getfile(selection))],
        },
    )


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserve existing benchmark evidence: {args.output}")
    manifest = json.loads(args.runtime_manifest.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Runtime manifest must be a JSON object")
    source = git_record(pathlib.Path(__file__).resolve().parents[1])
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    record = {
        "schema_version": 1,
        "status": "research-only",
        "source_reference": SOURCE_REFERENCE,
        "source_checkout": source,
        "runtime_manifest": file_record(args.runtime_manifest),
        "benchmark": file_record(__file__),
        "timing_utility": file_record(inspect.getfile(balanced_samples_us)),
        "argv": sys.argv,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": torch.cuda.get_device_capability(device),
        "warmup_pairs": args.warmup,
        "measured_pairs": args.repetitions,
        "seeds": [args.seed, args.seed + 1],
        "timing_mode": "eager CUDA events; alternating arm order",
        "ratio_direction": "candidate median microseconds / baseline median microseconds",
    }
    factories: dict[str, Callable] = {
        "query": query_case,
        "native-copy": native_copy_case,
        "selection": selection_case,
    }
    try:
        baseline, candidate, case = factories[args.case](args, device)
    except Exception as error:
        record["correctness"] = "failed before timing"
        record["error"] = {"type": type(error).__name__, "message": str(error)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        raise
    baseline_us, candidate_us = balanced_samples_us(
        baseline, candidate, warmup=args.warmup, iters=args.repetitions
    )
    record.update(case)
    record["baseline_us"] = baseline_us
    record["candidate_us"] = candidate_us
    record["baseline_median_us"] = statistics.median(baseline_us)
    record["candidate_median_us"] = statistics.median(candidate_us)
    record["candidate_over_baseline"] = (
        record["candidate_median_us"] / record["baseline_median_us"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_over_baseline": record["candidate_over_baseline"],
            }
        )
    )


if __name__ == "__main__":
    main()
