"""Compare native row-strided fused RMSNorm with a contiguous boundary copy."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from cuda.bindings import runtime as cudart

import b12x
from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


ARMS = ("native_strided", "boundary_copy", "packed_control")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _input_values(
    rows: int,
    hidden_size: int,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = torch.arange(
        rows * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(rows, hidden_size)
    inp = (torch.sin(offsets * 0.013 + rank * 0.17) * 0.25).to(torch.bfloat16)
    residual = (torch.cos(offsets * 0.007 + rows * 0.19) * 0.5).to(torch.bfloat16)
    weight = torch.linspace(
        0.5,
        1.5,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    return inp, residual, weight


def _split_view(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, hidden_size = values.shape
    combined = torch.full(
        (rows, hidden_size * 2),
        -7.0,
        dtype=values.dtype,
        device=values.device,
    )
    padding, residual = combined.split(hidden_size, dim=-1)
    residual.copy_(values)
    return combined, padding, residual


def _reference(
    inp: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reduced = inp.float().clone()
    dist.all_reduce(reduced)
    residual_fp32 = reduced + residual.float()
    inv_rms = torch.rsqrt(residual_fp32.square().mean(dim=-1, keepdim=True) + epsilon)
    return (residual_fp32 * inv_rms * weight.float()).to(inp.dtype), residual_fp32


def _graph_nodes(graph: torch.cuda.CUDAGraph) -> dict[str, int]:
    result, _, count = cudart.cudaGraphGetNodes(graph.raw_cuda_graph())
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaGraphGetNodes count failed: {result}")
    result, nodes, returned = cudart.cudaGraphGetNodes(graph.raw_cuda_graph(), count)
    if result != cudart.cudaError_t.cudaSuccess or returned != count:
        raise RuntimeError(
            f"cudaGraphGetNodes list failed: result={result}, returned={returned}"
        )
    counts: dict[str, int] = {}
    for node in nodes[:count]:
        result, node_type = cudart.cudaGraphNodeGetType(node)
        if result != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaGraphGetType failed: {result}")
        name = getattr(node_type, "name", str(node_type))
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def _error_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | int]:
    difference = actual.float() - reference.float()
    denominator = actual.float().norm() * reference.float().norm()
    cosine = float(
        (actual.float().flatten() @ reference.float().flatten() / denominator).item()
    )
    return {
        "max_abs": float(difference.abs().max().item()),
        "mismatch_count": int(torch.count_nonzero(actual != reference).item()),
        "cosine": cosine,
    }


def _capture_case(
    pool: PCIeOneshotAllReducePool,
    rows: int,
    hidden_size: int,
    rank: int,
    device: torch.device,
    epsilon: float,
) -> tuple[dict[str, torch.cuda.CUDAGraph], dict[str, Any], dict[str, Any]]:
    source_input, residual_values, weight = _input_values(
        rows, hidden_size, rank, device
    )
    inputs = {name: source_input.clone() for name in ARMS}
    _, native_padding, native_residual = _split_view(residual_values)
    _, boundary_padding, boundary_source = _split_view(residual_values)
    boundary_residual = residual_values.clone()
    packed_residual = residual_values.clone()
    residuals = {
        "native_strided": native_residual,
        "boundary_copy": boundary_residual,
        "packed_control": packed_residual,
    }
    outputs = {name: torch.empty_like(source_input) for name in ARMS}
    streams = {name: torch.cuda.Stream(device=device) for name in ARMS}
    graphs: dict[str, torch.cuda.CUDAGraph] = {}

    for name in ARMS:
        channel_id = f"graph:layout:{rows}:{name}"
        channel = pool.for_stream(streams[name], channel_id=channel_id)
        with torch.cuda.stream(streams[name]):
            channel.prepare_graph_fused_add_rms_norm(inputs[name])
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with (
            pool.capture(stream=streams[name], channel_id=channel_id),
            torch.cuda.graph(graph, stream=streams[name]),
        ):
            if name == "boundary_copy":
                boundary_residual.copy_(boundary_source)
            pool.all_reduce_fused_add_rms_norm(
                inputs[name],
                residuals[name],
                weight,
                epsilon,
                out=outputs[name],
                residual_out=residuals[name],
                stream=streams[name],
                channel_id=channel_id,
            )
        graphs[name] = graph

    expected_out, expected_residual_fp32 = _reference(
        source_input, residual_values, weight, epsilon
    )
    expected_residual_bf16 = expected_residual_fp32.to(torch.bfloat16)
    for name in ARMS:
        inputs[name].copy_(source_input)
    native_residual.copy_(residual_values)
    packed_residual.copy_(residual_values)
    boundary_source.copy_(residual_values)
    boundary_residual.fill_(0)
    native_padding.fill_(-7.0)
    boundary_padding.fill_(-7.0)
    for name in ARMS:
        graphs[name].replay()
        streams[name].synchronize()

    correctness = {
        name: {
            "output_vs_fp32_reference": _error_metrics(outputs[name], expected_out),
            "residual_vs_bf16_fp32_reference": _error_metrics(
                residuals[name], expected_residual_bf16
            ),
            "output_vs_packed_control": _error_metrics(
                outputs[name], outputs["packed_control"]
            ),
            "residual_vs_packed_control": _error_metrics(
                residuals[name], residuals["packed_control"]
            ),
        }
        for name in ARMS
    }
    correctness["native_strided"]["padding_mismatch_count"] = int(
        torch.count_nonzero(native_padding != -7.0).item()
    )
    correctness["boundary_copy"]["source_padding_mismatch_count"] = int(
        torch.count_nonzero(boundary_padding != -7.0).item()
    )

    allocation_deltas = {}
    for name in ARMS:
        before = torch.cuda.memory_allocated(device)
        for _ in range(3):
            graphs[name].replay()
        streams[name].synchronize()
        allocation_deltas[name] = torch.cuda.memory_allocated(device) - before

    metadata = {
        "inputs": inputs,
        "residuals": residuals,
        "streams": streams,
        "outputs": outputs,
        "weight": weight,
        "boundary_source": boundary_source,
        "native_padding": native_padding,
        "boundary_padding": boundary_padding,
    }
    result = {
        "rows": rows,
        "bytes": rows * hidden_size * torch.bfloat16.itemsize,
        "native_residual_stride": list(native_residual.stride()),
        "graph_nodes": {name: _graph_nodes(graph) for name, graph in graphs.items()},
        "replay_allocation_delta_bytes": allocation_deltas,
        "correctness": correctness,
    }
    return graphs, metadata, result


def _measure(
    graphs: dict[str, torch.cuda.CUDAGraph],
    streams: dict[str, torch.cuda.Stream],
    device: torch.device,
    samples: int,
    iterations: int,
    warmups: int,
) -> dict[str, list[float]]:
    for name in ARMS:
        for _ in range(warmups):
            graphs[name].replay()
        streams[name].synchronize()

    raw = {name: [] for name in ARMS}
    for sample in range(samples):
        order = list(ARMS[sample % len(ARMS) :]) + list(ARMS[: sample % len(ARMS)])
        if sample % 2:
            order.reverse()
        for name in order:
            dist.barrier()
            started = time.perf_counter()
            for _ in range(iterations):
                graphs[name].replay()
            streams[name].synchronize()
            elapsed_us = (time.perf_counter() - started) * 1e6 / iterations
            slowest = torch.tensor(elapsed_us, dtype=torch.float64, device=device)
            dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
            if dist.get_rank() == 0:
                raw[name].append(float(slowest.item()))
    return raw


def _worker(
    rank: int,
    world_size: int,
    port: int,
    rows: tuple[int, ...],
    hidden_size: int,
    samples: int,
    iterations: int,
    warmups: int,
    output: str,
) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    cases = []
    local_correctness = []
    for case_index, row_count in enumerate(rows):
        maximum_bytes = row_count * hidden_size * torch.bfloat16.itemsize
        channel_ids = tuple(f"graph:layout:{row_count}:{name}" for name in ARMS)
        pool = PCIeOneshotAllReducePool.from_process_group(
            process_group=dist.group.WORLD,
            device=device,
            max_input_bytes=maximum_bytes,
            max_size=maximum_bytes,
            max_concurrent_channels=len(channel_ids),
        )
        pool.prepare_channels(channel_ids)
        try:
            graphs, metadata, case = _capture_case(
                pool,
                row_count,
                hidden_size,
                rank,
                device,
                1e-6,
            )
            local_correctness.append(case["correctness"])
            if case_index == 0:
                b12x.freeze_kernel_resolution("fused residual layout benchmark")
            raw = _measure(
                graphs,
                metadata["streams"],
                device,
                samples,
                iterations,
                warmups,
            )
            if rank == 0:
                case["raw_slowest_rank_us"] = raw
                case["summary_slowest_rank_us"] = {
                    name: {
                        "median": statistics.median(values),
                        "p90": _percentile(values, 90.0),
                    }
                    for name, values in raw.items()
                }
                case["ratios"] = {
                    "native_strided_over_packed_control": (
                        statistics.median(raw["native_strided"])
                        / statistics.median(raw["packed_control"])
                    ),
                    "boundary_copy_over_native_strided": (
                        statistics.median(raw["boundary_copy"])
                        / statistics.median(raw["native_strided"])
                    ),
                }
                cases.append(case)
        finally:
            pool.close()

    gathered_correctness: list[Any] = [None] * world_size
    dist.all_gather_object(gathered_correctness, local_correctness)
    if rank == 0:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        record = {
            "schema_version": 1,
            "semantic_role": "TP8 fused all-reduce RMSNorm residual-layout comparison",
            "b12x_commit": revision,
            "world_size": world_size,
            "hidden_size": hidden_size,
            "dtype": "torch.bfloat16",
            "epsilon": 1e-6,
            "samples": samples,
            "iterations_per_sample": iterations,
            "warmups_per_arm": warmups,
            "kernel_resolution_frozen": b12x.kernel_resolution_frozen(),
            "transport_environment": {
                name: os.environ.get(name)
                for name in (
                    "B12X_PCIE_TP8_OWNER_REDUCE",
                    "B12X_PCIE_ONESHOT_PUSH",
                    "B12X_PCIE_FUSED_CTAS_PER_ROW",
                )
            },
            "cases": cases,
            "rank_correctness": gathered_correctness,
        }
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--rows", default="4,8,16")
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmups", type=int, default=100)
    args = parser.parse_args()
    rows = tuple(int(value) for value in args.rows.split(","))
    if args.world_size != 8:
        raise SystemExit("this comparison requires tensor parallelism 8")
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(
            f"need {args.world_size} CUDA devices, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(
            args.world_size,
            _free_port(),
            rows,
            args.hidden_size,
            args.samples,
            args.iterations,
            args.warmups,
            args.output,
        ),
        nprocs=args.world_size,
        join=True,
    )
    record = json.loads(Path(args.output).read_text())
    for case in record["cases"]:
        print(
            f"rows={case['rows']} native_us="
            f"{case['summary_slowest_rank_us']['native_strided']['median']:.3f} "
            f"boundary_us="
            f"{case['summary_slowest_rank_us']['boundary_copy']['median']:.3f} "
            f"packed_us="
            f"{case['summary_slowest_rank_us']['packed_control']['median']:.3f}"
        )


if __name__ == "__main__":
    main()
