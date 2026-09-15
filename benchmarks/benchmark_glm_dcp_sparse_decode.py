"""Compare GLM sparse-decode split plans for local and DCP query geometry.

Run on an idle GPU inside a committed image. This benchmark preserves a
2,048-column selection capacity while varying the number of valid candidates.
It compares the existing automatic split plan with explicit diagnostic plans;
it changes no serving policy. Every case checks packed-cache reference output,
natural-log LSE, and changed-input graph replay before collecting timings.
Each query geometry also runs with physical pages beyond the signed 32-bit
byte-address boundary.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import torch

from b12x._lib.runtime_control import (
    kernel_resolution_guard,
)
from b12x.attention import sparse_mla
from b12x.attention.sparse_mla._scratch import plan_sparse_mla_scratch
from b12x.attention._shared.mla.api import sparse_mla_decode_forward
from b12x.attention._shared.mla.kernel import LAST_DECODE_PLAN
from b12x.attention._shared.mla.reference import (
    pack_mla_kv_cache_reference,
    unpack_mla_kv_cache_reference,
)
from benchmarks.common import (
    device_provenance,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
    require_sm120,
    resolve_l2_flush_bytes,
    source_provenance,
)


def make_case(device, rows, heads, valid, high_pages):
    width, page_size, record_bytes = 2048, 64, 656
    generator = torch.Generator(device=device).manual_seed(1031 + rows + heads)
    latent = (
        torch.randn(
            (width, 1, 512), generator=generator, device=device, dtype=torch.float32
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    rope = (
        torch.randn(
            (width, 1, 64), generator=generator, device=device, dtype=torch.float32
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    packed = pack_mla_kv_cache_reference(latent, rope)
    first_page = (2**31 // (page_size * record_bytes) + 1) if high_pages else 0
    first_slot = first_page * page_size
    cache = torch.empty(
        (first_page + width // page_size, page_size, record_bytes),
        dtype=torch.uint8,
        device=device,
    )
    cache.view(-1, 1, record_bytes)[first_slot : first_slot + width].copy_(packed)
    q = (
        torch.randn(
            (rows, heads, 576), generator=generator, device=device, dtype=torch.float32
        )
        .mul_(0.25)
        .to(torch.bfloat16)
    )
    indices = (
        torch.arange(width, device=device, dtype=torch.int32).expand(rows, -1).clone()
    )
    indices.add_(first_slot)
    # Invalid tail slots point outside the pool if the active-length mask fails.
    indices[:, valid:].fill_(2**31 - 1)
    lengths = torch.full((rows,), valid, device=device, dtype=torch.int32)
    plan = plan_sparse_mla_scratch(
        sparse_mla.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=rows,
            max_width=width,
            softmax_scale=1 / math.sqrt(576),
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            head_dim=576,
            v_head_dim=512,
            mode="decode",
            max_batch=rows,
            max_chunks_per_row=32,
            page_size=page_size,
            return_lse=True,
            lse_scale="natural",
        )
    )
    scratch = [
        torch.empty(s.shape, dtype=s.dtype, device=s.device)
        for s in plan.scratch_specs()
    ]
    binding = plan.bind(
        scratch=scratch,
        q=q,
        kv_cache=cache,
        selected_indices=indices,
        cache_seqlens_int32=lengths,
        nsa_cache_seqlens_int32=lengths,
    )
    decoded = unpack_mla_kv_cache_reference(packed).squeeze(1).double()

    def reference():
        logits = torch.matmul(q.double(), decoded[:valid].T) / math.sqrt(576)
        return (
            torch.matmul(logits.softmax(-1), decoded[:valid, :512]),
            logits.logsumexp(-1),
        )

    def run(splits):
        return sparse_mla_decode_forward(
            binding=binding,
            kv_cache=cache,
            sm_scale=1 / math.sqrt(576),
            v_head_dim=512,
            return_lse=True,
            lse_scale="natural",
            forced_num_splits=splits,
        )

    return (
        q,
        run,
        reference,
        {
            "rows": rows,
            "heads": heads,
            "selection_capacity": width,
            "valid_candidates": valid,
            "first_physical_page": first_page,
            "first_physical_byte_offset": first_slot * record_bytes,
            "cache_shape": list(cache.shape),
            "cache_stride": list(cache.stride()),
        },
    )


def validate_and_capture(run, reference, splits):
    output = None
    for _ in range(3):
        output = run(splits)
    torch.cuda.synchronize()
    selected_plan = dict(LAST_DECODE_PLAN)
    if not selected_plan or selected_plan["num_tokens"] != output[0].shape[0]:
        raise RuntimeError("Sparse decode did not execute the unified kernel")
    expected = reference()
    check(output, expected)
    with kernel_resolution_guard("GLM sparse-decode split benchmark capture"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(splits)
    return graph, output, selected_plan


def check(output, expected):
    # FP8 query arithmetic differs from the BF16-query packed-cache oracle.
    # Split variants additionally compare directly to the automatic kernel.
    torch.testing.assert_close(output[0].double(), expected[0], rtol=0.02, atol=0.01)
    torch.testing.assert_close(output[1].double(), expected[1], rtol=0, atol=0.01)
    cosine = torch.nn.functional.cosine_similarity(
        output[0].double().flatten(), expected[0].flatten(), dim=0
    )
    if not torch.isfinite(cosine) or cosine < 0.999:
        raise RuntimeError(
            "Sparse decode output does not match the reference direction"
        )


def measure_case(device, rows, heads, valid, high_pages, args):
    q, run, reference, geometry = make_case(device, rows, heads, valid, high_pages)
    variants = {}
    for splits in args.splits:
        key = "automatic" if splits == 0 else str(splits)
        variants[key] = validate_and_capture(run, reference, splits or None)
    q.mul_(-0.5)
    expected = reference()
    snapshots = {}
    for key, (graph, output, _) in variants.items():
        graph.replay()
        torch.cuda.synchronize()
        check(output, expected)
        snapshots[key] = [tensor.clone() for tensor in output]
    # Snapshot allocations are outside the replay-allocation observation.
    allocated = torch.cuda.memory_allocated()
    pointers = {k: [t.data_ptr() for t in v[1]] for k, v in variants.items()}
    for key, (graph, output, _) in variants.items():
        for _ in range(30):
            graph.replay()
        torch.cuda.synchronize()
        if torch.cuda.memory_allocated() != allocated:
            raise RuntimeError("Graph replay changed allocated bytes")
        if pointers[key] != [tensor.data_ptr() for tensor in output]:
            raise RuntimeError("Graph output addresses changed")
        for tensor, snapshot in zip(output, snapshots[key], strict=True):
            if not torch.equal(tensor, snapshot):
                raise RuntimeError("Graph replay was not bitwise repeatable")
    baseline = snapshots["automatic"]
    for tensors in snapshots.values():
        for tensor, expected_tensor in zip(tensors, baseline, strict=True):
            torch.testing.assert_close(tensor, expected_tensor, rtol=0.002, atol=0.001)
    flush_bytes = resolve_l2_flush_bytes(0)
    flush = make_l2_flush_fn(True, flush_bytes)
    events = {key: [] for key in variants}
    for sample in range(args.samples):
        order = list(variants)
        if sample % 2:
            order.reverse()
        for key in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            flush()
            start.record()
            variants[key][0].replay()
            end.record()
            events[key].append((start, end))
    torch.cuda.synchronize()
    result = {}
    for key, pairs in events.items():
        samples = [start.elapsed_time(end) * 1000 for start, end in pairs]
        result[key] = {
            "selected_plan": variants[key][2],
            "samples_us": samples,
            "median_us": statistics.median(samples),
            "p90_us": statistics.quantiles(samples, n=10, method="inclusive")[8],
        }
    for row in result.values():
        row["ratio_over_automatic"] = (
            row["median_us"] / result["automatic"]["median_us"]
        )
    for key, (graph, output, _) in variants.items():
        graph.replay()
        torch.cuda.synchronize()
        for tensor, snapshot in zip(output, snapshots[key], strict=True):
            if not torch.equal(tensor, snapshot):
                raise RuntimeError("Timed replay changed the validated output")
    return {
        "geometry": geometry,
        "correctness_passed": True,
        "changed_input_replays": 30,
        "stable_addresses": True,
        "replay_allocation_delta_bytes": 0,
        "l2_flush_bytes": flush_bytes,
        "variants": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--splits", nargs="+", type=int, default=[0, 8, 16, 32])
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    if (
        args.samples < 20
        or 0 not in args.splits
        or any(s not in (0, 1, 2, 4, 8, 16, 32) for s in args.splits)
    ):
        parser.error(
            "Use at least 20 samples and supported split counts including 0 (automatic)"
        )
    if any(r not in (1, 4, 8, 16) for r in args.rows):
        parser.error("Rows must be one of 1, 4, 8, or 16")
    device = require_sm120()
    record = {
        "semantic_role": "GLM sparse decode split-plan comparison",
        "status": "unsupported",
        "argv": sys.argv,
        "source": source_provenance(),
        "device": device_provenance(device),
        "runtime_manifest": {
            "path": str(args.runtime_manifest.resolve()),
            "sha256": hashlib.sha256(args.runtime_manifest.read_bytes()).hexdigest(),
        },
        "gpu_before": nvidia_smi_gpu_mode_snapshot(device),
        "cases": [],
        "scope": "Standalone kernel and split merge; no transport or serving gate",
        "ratio_direction": "variant divided by automatic; smaller is faster",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for rows in args.rows:
            for heads, valid in ((8, 1024), (8, 2048), (32, 256), (32, 512)):
                for high_pages in (False, True):
                    row = measure_case(device, rows, heads, valid, high_pages, args)
                    record["cases"].append(row)
                    args.output.write_text(json.dumps(record, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                "geometry": row["geometry"],
                                "medians_us": {
                                    key: value["median_us"]
                                    for key, value in row["variants"].items()
                                },
                            }
                        ),
                        flush=True,
                    )
        record["status"] = "qualified"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        record["gpu_after"] = nvidia_smi_gpu_mode_snapshot(device)
        args.output.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
