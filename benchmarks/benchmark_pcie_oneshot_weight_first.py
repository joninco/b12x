"""Fused PCIe one-shot allreduce followed by the weight-first projection on
every GPU at once.

In a tensor-parallel decode step the router-gate plus shared-expert
gate_up projection (``[768, 6144]`` BF16, 9.4 MB) follows the
attention-output allreduce. With programmatic dependent launch the
projection's CTAs stage their weight bricks while the allreduce waits on the
fabric, and that staging traffic competes with the peers' reads of this
GPU's memory. This benchmark replays ``[one-shot fused add + RMSNorm ->
projection]`` graphs on every rank concurrently and reports the median
replay latency per variant, so the staging depth (``cp.async`` groups in
flight per CTA) can be chosen where the allreduce is not slowed.

Variants: ``none`` (allreduce alone), ``cublas`` (``torch.nn.functional.linear``),
``wf-serial`` (brick GEMV without the launch attribute), ``wf-d<depth>``
(brick GEMV with the attribute at staging depth ``depth``; 0 = unbounded).

The acceptance condition recorded for the production port is
``wf-d1`` below ``cublas`` at 4, 8 and 16 rows.

The one-shot transport must be configured as in the serving launch:
``B12X_PCIE_TP8_OWNER_REDUCE=0`` selects the transport the GLM-5.3 launches
use (the default, 1, is about twice as slow for these sizes in this
harness). The value in force is recorded in the JSON output.

Usage (8 GPUs):
  B12X_PCIE_TP8_OWNER_REDUCE=0 B12X_ONESHOT_ROWS=4,8,16 B12X_WF_DEPTHS=0,1,2,4 \\
      python benchmarks/benchmark_pcie_oneshot_weight_first.py [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from b12x.comm.pcie.pcie_oneshot import PCIeOneshotAllReducePool


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _median_latency(graph, device, warmup=50, iterations=300) -> float:
    dist.barrier()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(samples)


def _worker(rank: int, world_size: int, port: int, json_path: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size
    )
    from b12x.gemm import weight_first_gemv as wf

    hidden = int(os.getenv("B12X_WF_HIDDEN", "6144"))
    n_first = int(os.getenv("B12X_WF_N0", "256"))
    n_out = int(os.getenv("B12X_WF_N", "768"))
    rows_list = tuple(
        int(r) for r in os.getenv("B12X_ONESHOT_ROWS", "4,8,16").split(",")
    )
    depths = tuple(int(d) for d in os.getenv("B12X_WF_DEPTHS", "0,1,2,4").split(","))
    dtype = torch.bfloat16
    max_bytes = max(128 * 1024, max(rows_list) * hidden * 2)
    pool = PCIeOneshotAllReducePool.from_process_group(
        process_group=dist.group.WORLD,
        device=device,
        max_input_bytes=max_bytes,
        max_size=max_bytes,
        single_channel=True,
    )
    pool.for_stream()
    torch.manual_seed(11 + rank)

    def make_weights():
        w = (torch.randn(n_out, hidden, device=device) * 0.02).to(dtype)
        return [w[:n_first].clone(), w[n_first:].clone()]

    weight_cublas = (torch.randn(n_out, hidden, device=device) * 0.02).to(dtype)
    wf_by_depth = {
        d: wf.WeightFirstProjection(make_weights(), depth=d, pdl=True) for d in depths
    }
    wf_serial = wf.WeightFirstProjection(make_weights(), depth=0, pdl=False)
    norm_w = torch.ones(hidden, dtype=dtype, device=device)
    records = []
    try:
        if rank == 0:
            print("rows,variant,median_us", flush=True)
        for rows in rows_list:
            shape = (rows, hidden)
            x_in = torch.randn(shape, dtype=dtype, device=device) * 0.01
            residual = torch.randn(shape, dtype=dtype, device=device)
            out = torch.empty_like(x_in)
            residual_out = torch.empty_like(x_in)
            pool.prepare_graph_fused_add_rms_norm(x_in)

            def allreduce():
                pool.all_reduce_fused_add_rms_norm(
                    x_in, residual, norm_w, 1e-6, out=out, residual_out=residual_out
                )

            variants = [
                ("none", lambda: None),
                ("cublas", lambda: torch.nn.functional.linear(out, weight_cublas)),
                ("wf-serial", lambda: wf_serial(out)),
            ]
            for d in depths:
                variants.append((f"wf-d{d}", (lambda d=d: wf_by_depth[d](out))))
            for name, fn in variants:
                allreduce()
                fn()
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with pool.capture(), torch.cuda.graph(graph):
                    allreduce()
                    fn()
                us = _median_latency(graph, device)
                all_us = [None] * world_size
                dist.all_gather_object(all_us, us)
                if rank == 0:
                    print(f"{rows},{name},{us:.2f}", flush=True)
                    records.append(
                        {
                            "rows": rows,
                            "variant": name,
                            "median_us_rank0": us,
                            "median_us_per_rank": all_us,
                            "median_us_max_rank": max(all_us),
                        }
                    )
                del graph
                torch.cuda.synchronize(device)
        if rank == 0 and json_path:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "hidden": hidden,
                        "n": n_out,
                        "n0": n_first,
                        "rows": list(rows_list),
                        "depths": list(depths),
                        "world_size": world_size,
                        "env": {
                            key: os.environ.get(key)
                            for key in (
                                "B12X_PCIE_TP8_OWNER_REDUCE",
                                "B12X_PCIE_DMA_FP8",
                            )
                        },
                        "records": records,
                    },
                    fh,
                    indent=2,
                )
    finally:
        pool.close()
        dist.destroy_process_group()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--json", default="", help="write per-variant medians (all ranks) to this path"
    )
    ap.add_argument(
        "--world-size", type=int, default=int(os.getenv("B12X_ONESHOT_WORLD_SIZE", "8"))
    )
    args = ap.parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.device_count() < args.world_size:
        raise SystemExit(
            f"need {args.world_size} GPUs, found {torch.cuda.device_count()}"
        )
    mp.spawn(
        _worker,
        args=(args.world_size, _free_port(), args.json),
        nprocs=args.world_size,
        join=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
