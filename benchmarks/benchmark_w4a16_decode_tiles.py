#!/usr/bin/env python3
"""Measure packed W4A16 decode schedules with identical checkpoint inputs.

Research variants change compilation only while building independent public
execution plans and CUDA graphs. Timed replay uses those graphs directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from contextlib import contextmanager
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from b12x._lib.runtime_control import kernel_resolution_guard

import torch

from b12x.moe import fused_moe
from b12x.preparation import PreparationSession, PreparedCall
from b12x.moe._shared.kernels.w4a16 import kernel
from benchmarks.benchmark_moe import (
    MODEL_PROFILES,
    bench_events,
    build_model_spec,
    check_oracle_metrics,
    compare_to_reference,
    get_quant_mode_params,
    load_expert_weights,
    make_oracle_reference,
    make_profile_routed_inputs,
)
from benchmarks.common import (
    benchmark_provenance,
    make_l2_flush_fn,
    nvidia_smi_gpu_mode_snapshot,
)


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor.contiguous().view(torch.uint8).cpu().numpy()
    ).hexdigest()


@contextmanager
def variant(name):
    """Restore the reference capacity limit without changing kernel math."""
    limit = 16 if name == "reference" else kernel._PACKED_DECODE_WIDE_FC2_MAX_M
    with patch.object(kernel, "_PACKED_DECODE_WIDE_FC2_MAX_M", limit):
        yield


def check(actual, expected, label):
    assert bool(torch.isfinite(actual).all()), label
    assert bool(actual.count_nonzero()), label
    metrics = compare_to_reference(actual, expected)
    failures = check_oracle_metrics(
        label, metrics, len(actual), activation="silu", oracle_mode="w4a16"
    )
    assert not failures, failures
    return asdict(metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--variants", nargs="+", default=["reference", "implemented"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 314])
    parser.add_argument(
        "--routing",
        choices=["independent", "grouped4", "shared"],
        default="independent",
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert set(args.variants) <= {"reference", "implemented"}
    assert args.rounds > 0 and args.iterations > 0
    report = {
        "status": "running",
        "provenance": benchmark_provenance(),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(kernel.__file__))
        },
        "versions": {
            name: metadata.version(name)
            for name in ("torch", "triton", "nvidia-cutlass-dsl")
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "ratio_direction": "variant latency / reference latency; below one is faster",
        "input_description": "checkpoint weights; seeded synthetic activations and routing, held identical across variants",
        "cases": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    profile = MODEL_PROFILES["glm52"]
    spec = build_model_spec(args.model, profile, tp_size_override=8)
    report["geometry"] = asdict(spec)
    assert (spec.hidden_size, spec.I_tp, spec.num_experts, spec.top_k) == (
        6144,
        256,
        256,
        8,
    )
    print("Loading checkpoint expert layer", args.layer, flush=True)
    weights = load_expert_weights(
        args.model, spec, layer_idx=args.layer, checkpoint_family="glm"
    )
    params = get_quant_mode_params(weights, "shared", "w4a16")
    bundle = fused_moe.PackedWeights(
        w13=weights.w13_weight.clone(),
        w2=weights.w2_weight.clone(),
        w13_block_scales=weights.w13_blockscale_swizzled,
        w2_block_scales=weights.w2_blockscale_swizzled,
        w13_global_scales=params.g1_alphas,
        w2_global_scales=params.g2_alphas,
        input_scale=params.a1_gscale,
        intermediate_scale=params.a2_gscale,
    )
    report["checkpoint_hashes"] = {
        key: tensor_hash(getattr(bundle, key))
        for key in (
            "w13",
            "w2",
            "w13_block_scales",
            "w2_block_scales",
            "w13_global_scales",
            "w2_global_scales",
        )
    }
    weight_plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(
            format=weights.source_format, w13_layout=weights.w13_layout
        ),
        geometry=fused_moe.MoEGeometry(
            num_experts=256, hidden_size=6144, intermediate_size=256
        ),
        activation=fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.A16,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
        ),
        constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.MMA_PACKED
        ),
    )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=bundle)
    flush = make_l2_flush_fn(True, 128 << 20)
    for m in args.rows:
        for seed in args.seeds:
            print("Preparing", m, seed, flush=True)
            x, ids, route_weights = make_profile_routed_inputs(
                profile, weights, spec, m, seed, torch.device("cuda", 0)
            )
            if args.routing == "grouped4":
                ids = ids[::4].repeat_interleave(4, dim=0)[:m].contiguous()
            elif args.routing == "shared":
                ids = ids[:1].expand(m, -1).contiguous()
            expected = make_oracle_reference(
                "w4a16",
                "w4a16",
                x,
                weights,
                params,
                ids,
                route_weights,
                activation="silu",
            )
            case = {
                "rows": m,
                "seed": seed,
                "distinct_experts": int(ids.unique().numel()),
                "inputs": {
                    "activations": tensor_hash(x),
                    "ids": tensor_hash(ids),
                    "route_weights": tensor_hash(route_weights),
                },
                "arms": {},
                "gpu": [],
            }
            report["cases"].append(case)
            save()
            graphs = {}
            retained = []
            reference_out = None
            for name in args.variants:
                print("Compiling", m, seed, name, flush=True)
                with variant(name):
                    plan = fused_moe.plan_execution(
                        experts=experts,
                        capacity=fused_moe.ExecutionCapacity(
                            max_tokens=m, top_k=8, warmup_token_counts=(m,)
                        ),
                    )
                    def prepare_call(state):
                        specs = state.scratch.scratch_specs()
                        storage = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=x.device) for spec in specs)
                        prepared_output = torch.empty_like(x)
                        prepared_binding = state.bind(
                            scratch=storage, a=x, topk_ids=ids, topk_weights=route_weights,
                            output=prepared_output, input_scales_static=True,
                        )
                        return PreparedCall(
                            run=lambda: state.run(prepared_binding), output=prepared_output,
                            owners=(storage, prepared_binding),
                        )

                    # Variant overrides are process-local; compile in this process.
                    session = PreparationSession(device=x.device, autotune=False, compile_workers=0)
                    session.prepare((plan.request(name=f"{name}-rows-{m}", prepare_call=prepare_call),))
                    (scratch_spec,) = plan.scratch_specs()
                    scratch = torch.empty(
                        scratch_spec.shape, dtype=scratch_spec.dtype, device="cuda"
                    )
                    output = torch.empty_like(x)
                    binding = fused_moe.bind(
                        plan,
                        scratch=scratch,
                        a=x,
                        experts=experts,
                        topk_ids=ids,
                        topk_weights=route_weights,
                        output=output,
                        input_scales_static=True,
                    )
                    for _ in range(5):
                        fused_moe.run(binding=binding)
                    torch.cuda.synchronize()
                    arm = {
                        "oracle": check(output, expected, name),
                        "samples_us": {"cold": [], "warm": []},
                    }
                    if name == "reference":
                        reference_out = output.clone()
                    elif reference_out is not None:
                        arm["vs_reference"] = check(
                            output, reference_out, name + " versus reference"
                        )
                        arm["bitwise_equal"] = bool(torch.equal(output, reference_out))
                    compiled = []
                    for launch in kernel._FUSED_CACHE.values():
                        compiled.append(
                            {
                                key: getattr(launch, key)
                                for key in (
                                    "size_m",
                                    "weight_layout",
                                    "fc1_tile_k",
                                    "fc1_tile_n",
                                    "fc2_tile_k",
                                    "fc2_tile_n",
                                    "moe_block_size",
                                    "schedule_whole_tiles",
                                    "shared_memory_bytes",
                                    "cta_threads",
                                )
                            }
                        )
                    arm["compiled_cache"] = compiled
                    graph = torch.cuda.CUDAGraph()
                    with kernel_resolution_guard(
                        "expert decode schedule qualification"
                    ), torch.cuda.graph(graph):
                        fused_moe.run(binding=binding)
                    output.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    arm["poisoned_graph_oracle"] = check(
                        output, expected, name + " graph"
                    )
                case["arms"][name] = arm
                graphs[name] = (graph, output)
                retained.append((session, plan, scratch, binding))
                save()
            for graph, _output in graphs.values():
                for _ in range(20):
                    graph.replay()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            for round_idx in range(args.rounds):
                order = (
                    args.variants
                    if round_idx % 2 == 0
                    else list(reversed(args.variants))
                )
                case["gpu"].append(nvidia_smi_gpu_mode_snapshot())
                for name in order:
                    graph, output = graphs[name]
                    for mode in ("cold", "warm"):
                        samples = [
                            value * 1000
                            for value in bench_events(
                                graph.replay,
                                warmup=10,
                                iters=args.iterations,
                                l2_flush=flush if mode == "cold" else None,
                            )
                        ]
                        case["arms"][name]["samples_us"][mode].append(samples)
                save()
            assert torch.cuda.memory_allocated() == allocated
            case["replay_allocation_unchanged"] = True
            for name, (_graph, output) in graphs.items():
                case["arms"][name]["final_oracle"] = check(
                    output, expected, name + " final"
                )
                case["arms"][name]["median_us"] = {
                    mode: statistics.median([v for batch in samples for v in batch])
                    for mode, samples in case["arms"][name]["samples_us"].items()
                }
            for session, *_ in retained:
                session.close()
            print(
                "RESULT",
                m,
                seed,
                {name: arm["median_us"] for name, arm in case["arms"].items()},
                flush=True,
            )
            save()
    report["status"] = "complete; research-only"
    report["gpu_after"] = nvidia_smi_gpu_mode_snapshot()
    save()


if __name__ == "__main__":
    main()
