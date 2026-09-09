"""Capture standalone graph replay traces and inspect retained CPU launch scopes."""

import gzip
import hashlib
import json
import re
from pathlib import Path

import torch


def profile_replays(graph, output_dir: Path, rank: int, name: str, rows: int):
    """Record 34 replays and inspect scopes 2 through 31 for runtime allocation.

    Synchronization used to finish the capture is outside the replay scopes.
    Kernel inventory spans the full capture; timing gates use separate samples.
    """
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
    ) as profiler:
        for step in range(1, 35):
            with torch.profiler.record_function(f"dcp_transport_replay_{step}"):
                graph.replay()
        torch.cuda.synchronize()
    directory = output_dir / "replay-traces"
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}-rows-{rows}-rank-{rank}.json"
    profiler.export_chrome_trace(str(path))
    events = json.loads(path.read_text())["traceEvents"]
    trace = path.with_suffix(".json.gz")
    with gzip.open(trace, "wb") as stream:
        stream.write(path.read_bytes())
    path.unlink()
    scopes = [
        e
        for e in events
        if e.get("name", "").startswith("dcp_transport_replay_")
        and e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
        and 2 <= int(e["name"].rsplit("_", 1)[1]) <= 31
    ]
    assert len(scopes) == 30
    retained = [
        e
        for e in events
        if any(
            e.get("tid") == scope.get("tid")
            and e.get("pid") == scope.get("pid")
            and scope["ts"] <= e.get("ts", -1) < scope["ts"] + scope.get("dur", 0)
            for scope in scopes
        )
    ]
    allocations = [
        e
        for e in retained
        if re.search(
            r"(?:cuda|cu)(?:Malloc|Free|MemAlloc|MemFree)|^\[memory\]$",
            e.get("name", ""),
        )
    ]
    synchronizations = [e for e in retained if "Synchronize" in e.get("name", "")]
    eager = [
        e for e in retained if re.search(r"(?:cuda|cu)LaunchKernel", e.get("name", ""))
    ]
    graph_launches = [e for e in retained if "GraphLaunch" in e.get("name", "")]
    kernels = sorted(
        {e.get("name") for e in events if "kernel" in e.get("cat", "").lower()}
    )
    nccl = [name for name in kernels if "nccl" in name.lower()]
    device_to_host = [
        e for e in events if re.search(r"DtoH|Device to Host", e.get("name", ""))
    ]
    result = {
        "trace": {
            "path": str(trace),
            "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        },
        "retained_replay_scopes": 30,
        "graph_launch_count": len(graph_launches),
        "allocation_events": allocations,
        "host_synchronization_events": synchronizations,
        "eager_kernel_launch_events": eager,
        "kernel_names": kernels,
        "nccl_kernel_names": nccl,
        "device_to_host_copies": device_to_host,
        "passed": not (
            allocations or synchronizations or eager or nccl or device_to_host
        )
        and len(graph_launches) == 30
        and bool(kernels),
    }
    return result
