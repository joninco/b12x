"""Compare complete DCP exchanges on two concurrent four-GPU groups.

Status: research-only. Run via torchrun with eight ranks and no serving process.
Requires a runtime manifest for the committed image and writes raw samples per
rank. A CUDA graph batches operations to keep Python launch latency outside the
per-exchange timing resolution. Correctness, frozen resolution, stable output
addresses and replay allocation checks precede timing. Candidate outputs have
exact set semantics; their physical ordering is reported separately.

Example (inside a source-bound image with its Python environment activated)::

    torchrun --standalone --nproc-per-node=8 \
      -m benchmarks.benchmark_dcp_transport_matrix \
      --runtime-manifest /evidence/runtime-manifest.json \
      --output-dir /evidence/results --correctness-only

The implemented paths are the compiled symmetric-memory query/combine (with
an explicitly recorded query fallback if multicast is unavailable), PCIeDCPA2A
query/combine, rank-major candidate gather/selection, owner staging/selection/
direct result redistribution, and publication with peer consumers. Query-pull
timings include the actual sparse-attention consumer; subtract its matched
local-query case to estimate exchange overhead. Shared publication measures
independent query and candidate consumers with fixed attention indices.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import traceback
from datetime import UTC, datetime, timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from b12x._lib.runtime_control import (
    kernel_resolution_guard,
)
from benchmarks.dcp_transport.cases import attention_cases, candidate_case
from benchmarks.dcp_transport.prototype_cases import (
    publication_candidate_case,
    publication_combine_case,
    query_consumer_case,
)


def gpu_state():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,pstate,memory.used,"
            "clocks.sm,clocks.mem,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return {"captured_at": datetime.now(UTC).isoformat(), "csv": result.stdout}


def write_record(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def all_ranks(value):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def capture(case, repetitions):
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with case.capture(), torch.cuda.graph(graph):
        for _ in range(repetitions):
            case.run()
    return graph


def validate(case, repetitions):
    for _ in range(3):
        case.run()
    torch.cuda.synchronize()
    with kernel_resolution_guard("DCP transport graph capture"):
        graph = capture(case, repetitions)
    if case.refresh:
        case.refresh(1931)
    graph.replay()
    torch.cuda.synchronize()
    initial = [out.clone() for out in case.outputs]
    pointers = [out.data_ptr() for out in case.outputs]
    allocated = torch.cuda.memory_allocated()
    errors = []
    bitwise = True
    for replay in range(30):
        changed = replay == 15 and case.refresh is not None
        if changed:
            case.refresh(3911)
        graph.replay()
        torch.cuda.synchronize()
        try:
            case.check()
        except AssertionError as error:
            if not errors:
                errors.append(str(error))
        if changed:
            for snapshot, out in zip(initial, case.outputs, strict=True):
                snapshot.copy_(out)
        bitwise &= all(
            torch.equal(out, before)
            for out, before in zip(case.outputs, initial, strict=True)
        )
    allocation_delta = torch.cuda.memory_allocated() - allocated
    stable_pointers = pointers == [out.data_ptr() for out in case.outputs]
    if allocation_delta or not stable_pointers:
        errors.append("Replay changed allocated bytes or output pointers")
    if case.details.get("require_bitwise") and not bitwise:
        errors.append("Replay changed deterministic output order")
    return {
        "passed": not errors,
        "replays": 30,
        "errors": errors,
        "bitwise_output_order_repeatable": bitwise,
        "allocation_delta_bytes": allocation_delta,
        "stable_output_pointers": stable_pointers,
        "input_versions_after_capture": 2 if case.refresh else 0,
        "operations_per_replay": repetitions,
    }, graph


def measure(graph, args):
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    # Materialize event resources before collecting samples.
    start.record()
    end.record()
    end.synchronize()
    samples = []
    for _ in range(args.samples):
        dist.barrier()
        start.record()
        for _ in range(args.graphs_per_sample):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(
            start.elapsed_time(end) * 1000 / (args.repetitions * args.graphs_per_sample)
        )
    rank_samples = all_ranks(samples)
    maximum_samples = [max(column) for column in zip(*rank_samples, strict=True)]
    p90 = statistics.quantiles(maximum_samples, n=10, method="inclusive")[8]
    return {
        "unit": "microseconds_per_operation",
        "rank_samples": rank_samples,
        "max_rank_samples": maximum_samples,
        "max_rank_median": statistics.median(maximum_samples),
        "max_rank_p90": p90,
        "rank_medians": [statistics.median(s) for s in rank_samples],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--transports",
        nargs="+",
        choices=(
            "native",
            "b12x",
            "rank_major",
            "owner",
            "publication",
            "pull",
            "compiled_pull",
            "opaque_attention",
            "compiled_attention",
        ),
        default=["native", "b12x", "rank_major", "owner", "publication"],
    )
    parser.add_argument(
        "--consumer-splits", type=int, choices=(8, 11, 16, 32), default=32
    )
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--graphs-per-sample", type=int, default=8)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--profile-replays", action="store_true")
    args = parser.parse_args()
    if any(r not in range(1, 17) for r in args.rows):
        parser.error("rows must be in 1 through 16")
    if args.samples < 2 or args.repetitions < 1 or args.graphs_per_sample < 1:
        parser.error(
            "samples >= 2, repetitions >= 1 and graphs-per-sample >= 1 required"
        )
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        parser.error("This benchmark requires eight ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Gloo carries setup and timing coordination; NCCL exists only for the
    # compiled symmetric-memory workspace's device process group.
    dist.init_process_group("gloo", timeout=timedelta(minutes=10))
    for peers in (list(range(4)), list(range(4, 8))):
        cpu = dist.new_group(peers, backend="gloo")
        gpu = dist.new_group(peers, backend="nccl")
        if rank in peers:
            group, gpu_group = cpu, gpu
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"rank-{rank}.json"
    record = {
        "schema": "b12x.dcp_transport_matrix.v1",
        "status": "research-only",
        "started_at": datetime.now(UTC).isoformat(),
        "rank": rank,
        "dcp_rank": rank % 4,
        "command": [sys.executable, *sys.argv],
        "runtime_manifest": {
            "path": str(args.runtime_manifest),
            "sha256": hashlib.sha256(args.runtime_manifest.read_bytes()).hexdigest(),
        },
        "measurement": {
            "samples": args.samples,
            "operations_per_graph": args.repetitions,
            "graphs_per_sample": args.graphs_per_sample,
            "correctness_only": args.correctness_only,
        },
        "gpu_before": gpu_state(),
        "cases": [],
        "selection_complete": False,
        "consumer_splits": args.consumer_splits,
    }
    write_record(path, record)
    failed = False
    try:
        for rows in args.rows:
            # Rotate candidate order across shapes to avoid a constant first
            # candidate receiving all cold-device measurements.
            shift = args.rows.index(rows) % len(args.transports)
            order = args.transports[shift:] + args.transports[:shift]
            for kind in order:
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "rows": rows,
                            "transport": kind,
                            "state": "construct",
                        }
                    ),
                    flush=True,
                )
                if kind == "publication":
                    cases = [
                        query_consumer_case(
                            mode,
                            group,
                            device,
                            rows,
                            rank % 4,
                            splits=args.consumer_splits,
                        )
                        for mode in (
                            "local_consumer",
                            "published_consumer",
                            "shared_consumers",
                        )
                    ]
                    cases.append(
                        publication_combine_case(group, device, rows, rank % 4)
                    )
                    cases.append(
                        publication_candidate_case(group, device, rows, rank % 4)
                    )
                elif kind in ("pull", "compiled_pull"):
                    cases = [
                        publication_candidate_case(
                            group,
                            device,
                            rows,
                            rank % 4,
                            packaged=True,
                            compiled=kind == "compiled_pull",
                        )
                    ]
                else:
                    cases = (
                        attention_cases(kind, group, gpu_group, device, rows, rank % 4)
                        if kind
                        in ("native", "b12x", "opaque_attention", "compiled_attention")
                        else [candidate_case(kind, group, device, rows, rank % 4)]
                    )
                execution_stream = None
                if kind in (
                    "opaque_attention",
                    "compiled_attention",
                    "pull",
                    "compiled_pull",
                ):
                    execution_stream = torch.cuda.Stream(device=device)
                    execution_stream.wait_stream(torch.cuda.current_stream())
                    for case in cases:
                        case.details["separate_construction_and_execution_streams"] = (
                            True
                        )
                        case.details["construction_stream_id"] = int(
                            torch.cuda.current_stream().cuda_stream
                        )
                        case.details["execution_stream_id"] = int(
                            execution_stream.cuda_stream
                        )
                try:
                    with torch.cuda.stream(execution_stream):
                        for case in cases:
                            result = {
                                "name": case.name,
                                "operation": case.operation,
                                "rows": rows,
                                "details": case.details,
                                "gpu_before_capture": gpu_state(),
                            }
                            validation, graph = validate(case, args.repetitions)
                            result["correctness"] = all_ranks(validation)
                            passed = all(c["passed"] for c in result["correctness"])
                            failed |= not passed
                            if passed and not args.correctness_only:
                                result["timing"] = measure(graph, args)
                            if passed and args.profile_replays:
                                from benchmarks.dcp_transport.replay_profile import (
                                    profile_replays,
                                )

                                result["replay_profile"] = profile_replays(
                                    graph, args.output_dir, rank, case.name, rows
                                )
                                profile_ok = all(
                                    all_ranks(result["replay_profile"]["passed"])
                                )
                                failed |= not profile_ok
                            del graph
                            result["gpu_after_replays"] = gpu_state()
                            record["cases"].append(result)
                            write_record(path, record)
                            print(
                                json.dumps(
                                    {
                                        "rank": rank,
                                        "rows": rows,
                                        "case": case.name,
                                        "correctness": passed,
                                    }
                                ),
                                flush=True,
                            )
                finally:
                    for case in reversed(cases):
                        case.close()
        record["completed"] = True
        record["all_correctness_passed"] = not failed
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["gpu_after"] = gpu_state()
        record["ended_at"] = datetime.now(UTC).isoformat()
        write_record(path, record)
    dist.barrier()
    dist.destroy_process_group()
    if failed:
        raise SystemExit("At least one transport failed correctness; see rank records")


if __name__ == "__main__":
    main()
