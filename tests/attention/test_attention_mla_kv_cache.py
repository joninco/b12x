from __future__ import annotations

import pytest
import torch

from b12x.attention import sparse_mla
from b12x.attention._shared.mla.kernel import run_unified_decode
from b12x.attention._shared.mla.kv_cache import (
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache,
    concat_and_cache_nvfp4_mla_fp8_rope,
)
from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat
from b12x.attention.sparse_mla._scratch import (
    B12XSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)

from tests._reference.helpers import (
    dequantize_nvfp4_mla_nope,
    require_b12x,
)


_RECORD_BYTES = 368
_NOPE_BYTES = 256
_GROUP_SCALES_OFFSET = 256
_ROPE_SCALE_OFFSET = 288
_LATENT_SCALE_OFFSET = 292
_PAD_OFFSET = 296
_ROPE_OFFSET = 304
_PAGE_SIZE = 64
_HEADS = 64
_HEAD_DIM = 576
_V_HEAD_DIM = 512
_SENTINEL = 0xA5


def test_nvfp4_mla_fp8_rope_writer_is_public() -> None:
    assert (
        sparse_mla.concat_and_cache_nvfp4_mla_fp8_rope
        is concat_and_cache_nvfp4_mla_fp8_rope
    )


@pytest.fixture(scope="module", autouse=True)
def _isolate_writer_kernel_cache():
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache()
    yield
    clear_nvfp4_mla_fp8_rope_kv_cache_kernel_cache()


def _valid_cpu_writer_args() -> list[torch.Tensor]:
    return [
        torch.empty((2, _V_HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((2, _HEAD_DIM - _V_HEAD_DIM), dtype=torch.bfloat16),
        torch.empty((2, _PAGE_SIZE, _RECORD_BYTES), dtype=torch.uint8),
        torch.arange(2, dtype=torch.int64),
    ]


def _invalid_writer_args(case: str) -> tuple[torch.Tensor, ...]:
    args = _valid_cpu_writer_args()
    if case == "kv_c_shape":
        args[0] = torch.empty((2, _V_HEAD_DIM - 1), dtype=torch.bfloat16)
    elif case == "k_pe_shape":
        args[1] = torch.empty((2, _HEAD_DIM - _V_HEAD_DIM - 1), dtype=torch.bfloat16)
    elif case == "kv_c_dtype":
        args[0] = torch.empty((2, _V_HEAD_DIM), dtype=torch.float32)
    elif case == "k_pe_dtype":
        args[1] = torch.empty((2, _HEAD_DIM - _V_HEAD_DIM), dtype=torch.float16)
    elif case == "cache_shape":
        args[2] = torch.empty((2, _PAGE_SIZE, _RECORD_BYTES - 1), dtype=torch.uint8)
    elif case == "cache_dtype":
        args[2] = torch.empty((2, _PAGE_SIZE, _RECORD_BYTES), dtype=torch.bfloat16)
    elif case == "zero_num_blocks":
        args[2] = torch.empty((0, _PAGE_SIZE, _RECORD_BYTES), dtype=torch.uint8)
    elif case == "zero_block_size":
        args[2] = torch.empty((2, 0, _RECORD_BYTES), dtype=torch.uint8)
    elif case == "slot_dtype":
        args[3] = torch.arange(2, dtype=torch.int32)
    elif case == "slot_layout":
        args[3] = torch.arange(4, dtype=torch.int64)[::2]
    elif case == "short_source":
        args[0] = args[0][:1]
    elif case == "kv_c_inner_layout":
        args[0] = torch.empty((2, 2 * _V_HEAD_DIM), dtype=torch.bfloat16)[:, ::2]
    elif case == "cache_inner_layout":
        args[2] = torch.empty((2, _PAGE_SIZE, 2 * _RECORD_BYTES), dtype=torch.uint8)[
            ..., ::2
        ]
    elif case == "kv_c_row_alignment":
        args[0] = torch.empty((2, _V_HEAD_DIM + 1), dtype=torch.bfloat16)[
            :, :_V_HEAD_DIM
        ]
    elif case == "k_pe_row_alignment":
        args[1] = torch.empty((2, _HEAD_DIM - _V_HEAD_DIM + 1), dtype=torch.bfloat16)[
            :, : _HEAD_DIM - _V_HEAD_DIM
        ]
    elif case == "cache_record_alignment":
        args[2] = torch.empty((2, _PAGE_SIZE, _RECORD_BYTES + 1), dtype=torch.uint8)[
            ..., :_RECORD_BYTES
        ]
    elif case == "cache_base_alignment":
        numel = 2 * _PAGE_SIZE * _RECORD_BYTES
        args[2] = torch.empty(numel + 8, dtype=torch.uint8)[8:].view(
            2, _PAGE_SIZE, _RECORD_BYTES
        )
    elif case != "cpu_device":
        raise AssertionError(f"unknown invalid writer case {case}")
    return tuple(args)


@pytest.mark.parametrize(
    ("case", "error", "match"),
    [
        ("kv_c_shape", ValueError, r"kv_c must be \(num_tokens, 512\)"),
        ("k_pe_shape", ValueError, r"k_pe must be \(num_tokens, 64\)"),
        ("kv_c_dtype", TypeError, "kv_c must be bf16/f16"),
        ("k_pe_dtype", TypeError, "must match kv_c dtype"),
        ("cache_shape", ValueError, "kv_cache must be"),
        ("cache_dtype", TypeError, "kv_cache must be uint8"),
        ("zero_num_blocks", ValueError, "num_blocks.*positive"),
        ("zero_block_size", ValueError, "block_size.*positive"),
        ("slot_dtype", TypeError, "slot_mapping must be a 1-D int64 tensor"),
        ("slot_layout", ValueError, "slot_mapping must be contiguous"),
        ("short_source", ValueError, "must cover slot_mapping"),
        ("kv_c_inner_layout", ValueError, "must be innermost-contiguous"),
        ("cache_inner_layout", ValueError, "must be innermost-contiguous"),
        ("kv_c_row_alignment", ValueError, "must be 4-byte aligned"),
        ("k_pe_row_alignment", ValueError, "k_pe.*4-byte aligned"),
        ("cache_record_alignment", ValueError, "must be 16-byte aligned"),
        ("cache_base_alignment", ValueError, "must be 16-byte aligned"),
        ("cpu_device", ValueError, "all tensors must be on CUDA"),
    ],
)
def test_writer_rejects_invalid_dtype_shape_device_and_layout(
    case: str,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        concat_and_cache_nvfp4_mla_fp8_rope(*_invalid_writer_args(case))


def _make_exactly_quantizable_inputs(
    *,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    e2m1_values = torch.tensor(
        [
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
            2.0,
            -2.0,
        ],
        dtype=torch.float32,
    )
    e2m1_codes = torch.tensor(
        [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 4, 12],
        dtype=torch.uint8,
    )
    group_scales = torch.empty((num_tokens, 32), dtype=torch.float32)
    nope = torch.empty((num_tokens, 32, 16), dtype=torch.float32)
    codes = torch.empty((num_tokens, 32, 16), dtype=torch.uint8)
    for token in range(num_tokens):
        for group in range(32):
            scale = 2.0 ** ((token + group) % 4 - 2)
            shift = (5 * token + 3 * group) % 16
            group_scales[token, group] = scale
            nope[token, group] = torch.roll(e2m1_values, shifts=shift) * scale
            codes[token, group] = torch.roll(e2m1_codes, shifts=shift)

    packed_nope = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(
        num_tokens, _NOPE_BYTES
    )

    e4m3_values = torch.tensor(
        [
            448.0,
            -448.0,
            416.0,
            -416.0,
            240.0,
            -240.0,
            120.0,
            -120.0,
            6.0,
            -6.0,
            1.5,
            -1.5,
            0.5,
            -0.5,
            0.0,
            2.0,
        ],
        dtype=torch.float32,
    ).repeat(4)
    rope_scales = torch.tensor(
        [2.0 ** (token - 4) for token in range(num_tokens)],
        dtype=torch.float32,
    )
    rope = torch.stack(
        [
            torch.roll(e4m3_values, shifts=7 * token) * rope_scales[token]
            for token in range(num_tokens)
        ]
    )
    return (
        nope.reshape(num_tokens, _V_HEAD_DIM).to(device=device, dtype=dtype),
        rope.to(device=device, dtype=dtype),
        packed_nope.to(device=device),
        group_scales.to(device=device),
        rope_scales.to(device=device),
    )


def _dequantize_records(
    records: torch.Tensor,
    *,
    per_token_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nope, group_scales = dequantize_nvfp4_mla_nope(
        records,
        nope_bytes=_NOPE_BYTES,
        group_scales_offset=_GROUP_SCALES_OFFSET,
        group_scales_end=_ROPE_SCALE_OFFSET,
        latent_scale_offset=(_LATENT_SCALE_OFFSET if per_token_scale else None),
    )

    rope_scales = (
        records[:, _ROPE_SCALE_OFFSET:_LATENT_SCALE_OFFSET]
        .contiguous()
        .view(torch.float32)
        .reshape(-1)
    )
    rope_q = (
        records[:, _ROPE_OFFSET:_RECORD_BYTES]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    rope = rope_q * rope_scales.unsqueeze(-1)
    return nope, group_scales, rope, rope_scales


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@torch.inference_mode()
def test_writer_preserves_skipped_slots_and_writes_the_record_abi(
    dtype: torch.dtype,
) -> None:
    device = require_b12x()
    num_tokens = 6
    kv_c, k_pe, expected_packed, expected_group_scales, expected_rope_scales = (
        _make_exactly_quantizable_inputs(
            num_tokens=num_tokens,
            dtype=dtype,
            device=device,
        )
    )

    # The cache view has a full guard page on each side. The capacity slot
    # targets the first trailing guard record without an upper-bound check. The
    # huge slot's low 32 bits name an otherwise untouched in-cache record, so a
    # check performed only after Int32 narrowing is also observable.
    backing = torch.full(
        (5, _PAGE_SIZE, _RECORD_BYTES),
        _SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    cache = backing[1:4]
    capacity = cache.shape[0] * cache.shape[1]
    slot_mapping = torch.tensor(
        [0, -1, capacity, 65, 2**40 + 17, 130],
        dtype=torch.int64,
        device=device,
    )
    if dtype == torch.float16:
        concat_and_cache_nvfp4_mla_fp8_rope(
            kv_c,
            k_pe,
            cache,
            slot_mapping,
            scale=torch.tensor([37.0], dtype=torch.float32, device=device),
        )
    else:
        concat_and_cache_nvfp4_mla_fp8_rope(kv_c, k_pe, cache, slot_mapping)
    torch.cuda.synchronize(device)

    changed_records = (
        (backing != _SENTINEL)
        .any(dim=-1)
        .reshape(-1)
        .nonzero(as_tuple=False)
        .reshape(-1)
    )
    expected_changed = torch.tensor(
        [_PAGE_SIZE, 2 * _PAGE_SIZE + 1, 3 * _PAGE_SIZE + 2],
        dtype=torch.int64,
        device=device,
    )
    assert torch.equal(changed_records, expected_changed)

    live_slots = torch.tensor([0, 65, 130], dtype=torch.int64, device=device)
    live_tokens = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    records = cache.reshape(-1, _RECORD_BYTES).index_select(0, live_slots)

    # These byte ranges are the public record ABI consumed by the real readers.
    assert torch.equal(
        records[:, :_NOPE_BYTES], expected_packed.index_select(0, live_tokens)
    )
    assert (
        torch.count_nonzero(records[:, _LATENT_SCALE_OFFSET:_ROPE_OFFSET]).item() == 0
    )

    nope, group_scales, rope, rope_scales = _dequantize_records(records)
    torch.testing.assert_close(
        group_scales,
        expected_group_scales.index_select(0, live_tokens),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        rope_scales,
        expected_rope_scales.index_select(0, live_tokens),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        nope,
        kv_c.index_select(0, live_tokens).float(),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        rope,
        k_pe.index_select(0, live_tokens).float(),
        rtol=0.0,
        atol=1e-5,
    )


def _make_written_reader_case(
    *,
    rows: int,
    topk: int,
    seed: int,
    device: torch.device,
    per_token_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    kv_c = torch.randn(
        (topk, _V_HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    k_pe = torch.randn(
        (topk, _HEAD_DIM - _V_HEAD_DIM),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    q = torch.randn(
        (rows, _HEADS, _HEAD_DIM), generator=generator, dtype=torch.float32
    ).to(device=device, dtype=torch.bfloat16)
    indices = torch.stack(
        [torch.randperm(topk, generator=generator) for _ in range(rows)]
    ).to(device=device, dtype=torch.int32)

    num_blocks = (topk + _PAGE_SIZE - 1) // _PAGE_SIZE
    cache = torch.full(
        (num_blocks, _PAGE_SIZE, _RECORD_BYTES),
        _SENTINEL,
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(topk, dtype=torch.int64, device=device)
    concat_and_cache_nvfp4_mla_fp8_rope(
        kv_c,
        k_pe,
        cache,
        slots,
        per_token_scale=per_token_scale,
    )
    return q, cache, indices


def _reference_attention_from_records(
    *,
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sm_scale: float,
    per_token_scale: bool = False,
) -> torch.Tensor:
    nope, _, rope, _ = _dequantize_records(
        cache.reshape(-1, _RECORD_BYTES),
        per_token_scale=per_token_scale,
    )
    keys = torch.cat((nope, rope), dim=-1)
    rows = []
    for row, length in enumerate(lengths.cpu().tolist()):
        selected = indices[row, :length].long()
        selected_keys = keys.index_select(0, selected)
        selected_values = nope.index_select(0, selected)
        scores = q[row].float() @ selected_keys.T
        probabilities = torch.softmax(scores * sm_scale, dim=-1)
        rows.append(probabilities @ selected_values)
    return torch.stack(rows)


def _assert_reader_matches_dequantized_records(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_f = actual.float()
    max_abs = (actual_f - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1), expected.reshape(-1), dim=0
    ).item()
    assert max_abs < 0.03, f"max absolute error {max_abs:.8f} exceeded 0.03"
    assert cosine > 0.999, f"cosine similarity {cosine:.8f} did not exceed 0.999"


@pytest.mark.parametrize(
    "per_token_scale",
    [False, True],
    ids=["static", "dynamic-token"],
)
@torch.inference_mode()
def test_writer_records_feed_production_head_multisplit_decode(
    per_token_scale: bool,
) -> None:
    device = require_b12x()
    topk = 129
    q, cache, indices = _make_written_reader_case(
        rows=1,
        topk=topk,
        seed=1301,
        device=device,
        per_token_scale=per_token_scale,
    )
    lengths = torch.full((1,), topk, dtype=torch.int32, device=device)
    sm_scale = _HEAD_DIM**-0.5

    caps = B12XSparseMLAScratchCaps(
        softmax_scale=1.0,
        device=device,
        dtype=torch.bfloat16,
        kv_dtype=torch.uint8,
        num_q_heads=_HEADS,
        max_q_rows=1,
        max_batch=1,
        max_width=topk,
        max_kv_rows=topk,
        head_dim=_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        max_chunks_per_row=8,
        page_size=_PAGE_SIZE,
    )
    plan = plan_sparse_mla_scratch(caps)
    (scratch_spec,) = plan.scratch_specs()
    scratch_storage = torch.zeros(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )
    binding = plan.bind(
        scratch=scratch_storage,
        q=q,
        selected_indices=indices,
        cache_seqlens_int32=lengths,
        nsa_cache_seqlens_int32=lengths,
    )

    actual = run_unified_decode(
        q_all=q,
        swa_k_cache=cache,
        swa_indices=indices,
        swa_topk_lengths=lengths,
        workspace=binding.scratch,
        sm_scale=sm_scale,
        swa_page_size=_PAGE_SIZE,
        forced_num_splits=2,
        scale_format_override=ScaleFormat.NVFP4_E4M3,
        fp8_rope_override=None,
        latent_scale_per_token=per_token_scale,
    )
    expected = _reference_attention_from_records(
        q=q,
        cache=cache,
        indices=indices,
        lengths=lengths,
        sm_scale=sm_scale,
        per_token_scale=per_token_scale,
    )
    torch.cuda.synchronize(device)
    _assert_reader_matches_dequantized_records(actual, expected)


@pytest.mark.parametrize(
    "rows,heads,high_page_ids",
    [(1, 8, False), (4, 32, False), (16, 32, False), (4, 32, True)],
)
@pytest.mark.parametrize("mode", ["decode", "extend"])
@torch.inference_mode()
def test_nvfp4_natural_lse_matches_quantized_records(
    rows: int, heads: int, high_page_ids: bool, mode: str
) -> None:
    """DCP's 32-head partials must carry the LSE of their quantized KV rows."""
    device = require_b12x()
    topk = 129
    q, compact_cache, indices = _make_written_reader_case(
        rows=rows, topk=topk, seed=4103, device=device
    )
    q = q[:, :heads].contiguous()
    lengths = torch.tensor(
        ([1, 3, 64, topk] * 4)[:rows], dtype=torch.int32, device=device
    )
    cache = compact_cache
    physical_indices = torch.full((rows, 2048), -1, dtype=torch.int32, device=device)
    physical_indices[:, :topk].copy_(indices)
    if high_page_ids:
        # A small live tail crosses the signed 32-bit byte-offset boundary.
        page_base = (1 << 31) // (_PAGE_SIZE * _RECORD_BYTES) + 1
        cache = torch.empty(
            (page_base + compact_cache.shape[0], _PAGE_SIZE, _RECORD_BYTES),
            dtype=torch.uint8,
            device=device,
        )
        cache[page_base:].copy_(compact_cache)
        physical_indices[:, :topk] += page_base * _PAGE_SIZE
    sm_scale = 192**-0.5
    plan = sparse_mla.plan(
        sparse_mla.Caps(
            device=device,
            num_q_heads=heads,
            max_q_rows=16,
            max_batch=16,
            max_width=2048,
            max_chunks_per_row=32,
            page_size=_PAGE_SIZE,
            softmax_scale=sm_scale,
            dtype=torch.bfloat16,
            kv_dtype=torch.uint8,
            scale_format=ScaleFormat.NVFP4_E4M3,
            fp8_rope=True,
            return_lse=True,
            lse_scale="natural",
            mode=mode,
        )
    )
    spec = plan.scratch_specs()[0]
    storage = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    binding = sparse_mla.bind(
        plan,
        scratch=storage,
        q=q,
        kv_cache=cache,
        selected_indices=physical_indices,
        cache_lengths=torch.full_like(lengths, topk),
        selected_lengths=lengths,
    )
    output, lse = sparse_mla.run(binding)
    eager_output, eager_lse = output.clone(), lse.clone()
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        sparse_mla.run(binding)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, lse = sparse_mla.run(binding)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize(device)

    nope, _, rope, _ = _dequantize_records(compact_cache.reshape(-1, _RECORD_BYTES))
    keys = torch.cat((nope, rope), dim=-1).double()
    expected_output, expected_lse = [], []
    for row, length in enumerate(lengths.cpu().tolist()):
        selected = indices[row, :length].long()
        scores = q[row].double() @ keys.index_select(0, selected).T * sm_scale
        expected_lse.append(torch.logsumexp(scores, dim=-1))
        expected_output.append(
            torch.softmax(scores, dim=-1) @ nope.index_select(0, selected).double()
        )
    expected_output = torch.stack(expected_output).float()
    expected_lse = torch.stack(expected_lse).float()
    for actual_output, actual_lse in ((eager_output, eager_lse), (output, lse)):
        _assert_reader_matches_dequantized_records(actual_output, expected_output)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-3, atol=1e-2)


@pytest.mark.parametrize(
    "per_token_scale",
    [False, True],
    ids=["static", "dynamic-token"],
)
@torch.inference_mode()
def test_writer_records_feed_production_head_multitile_prefill_mg(
    per_token_scale: bool,
) -> None:
    device = require_b12x()
    topk = 129
    q, cache, indices = _make_written_reader_case(
        rows=2,
        topk=topk,
        seed=2302,
        device=device,
        per_token_scale=per_token_scale,
    )
    lengths = torch.tensor([topk, 65], dtype=torch.int32, device=device)
    sm_scale = _HEAD_DIM**-0.5

    actual, _ = run_unified_prefill_mg(
        q=q,
        kv_cache=cache,
        topk_indices=indices,
        topk_length=lengths,
        sm_scale=sm_scale,
        page_block_size=_PAGE_SIZE,
        compute_mode=ComputeMode.BF16,
        mg_n_hg=2,
        model_type=ModelType.GLM_NSA,
        scale_format=ScaleFormat.NVFP4_E4M3,
        fp8_rope=True,
        latent_scale_per_token=per_token_scale,
    )
    expected = _reference_attention_from_records(
        q=q,
        cache=cache,
        indices=indices,
        lengths=lengths,
        sm_scale=sm_scale,
        per_token_scale=per_token_scale,
    )
    torch.cuda.synchronize(device)
    _assert_reader_matches_dequantized_records(actual, expected)


def _fp8_dsa_record_reference(latent: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    """CPU byte oracle for group-128 E4M3 latent plus unchanged BF16 RoPE."""
    groups = latent.cpu().float().reshape(-1, 4, 128)
    scales = (groups.abs().amax(-1) / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    quant = (groups / scales[..., None]).to(torch.float8_e4m3fn)
    return torch.cat(
        (
            quant.reshape(-1, 512).view(torch.uint8),
            scales.contiguous().view(torch.uint8),
            rope.cpu().contiguous().view(torch.uint8),
        ),
        dim=1,
    )


def test_fp8_dsa_record_oracle_preserves_rope_and_zero_scale_floor():
    latent = torch.zeros((2, 512), dtype=torch.bfloat16)
    latent[1, 0] = 448
    rope = torch.arange(128).reshape(2, 64).to(torch.bfloat16)
    records = _fp8_dsa_record_reference(latent, rope)
    assert records.shape == (2, 656)
    assert torch.equal(records[:, 528:], rope.view(torch.uint8))
    scales = records[:, 512:528].contiguous().view(torch.float32)
    assert scales[1, 0] == 1
    assert torch.all(scales[0] == torch.finfo(torch.float32).tiny)
    assert records[1, 0] == torch.tensor(448.0).to(torch.float8_e4m3fn).view(
        torch.uint8
    )


@pytest.mark.parametrize("case", ["latent", "rope", "cache", "slot", "rows", "device"])
def test_fp8_dsa_writer_rejects_invalid_contract_on_cpu(case):
    from b12x.attention._shared.mla.kv_cache import concat_and_cache_fp8_ds_mla

    latent = torch.empty((2, 512), dtype=torch.bfloat16)
    rope = torch.empty((2, 64), dtype=torch.bfloat16)
    cache = torch.empty((1, 4, 656), dtype=torch.uint8)
    slots = torch.arange(2, dtype=torch.int64)
    rows = 2
    if case == "latent":
        latent = latent.float()
    elif case == "rope":
        rope = rope[:, :32]
    elif case == "cache":
        cache = cache[..., :655]
    elif case == "slot":
        slots = slots.int()
    elif case == "rows":
        rows = 3
    with pytest.raises(ValueError):
        concat_and_cache_fp8_ds_mla(latent, rope, cache, slots, num_tokens=rows)


@pytest.mark.parametrize("case", ["out", "rows", "rank_shape", "boundaries", "device"])
def test_ckv_current_chunk_mapping_rejects_invalid_contract_on_cpu(case):
    from b12x.attention._shared.mla.kv_cache import map_ckv_current_chunk_slots

    starts = torch.tensor([0, 2])
    requests = torch.tensor([0, 0])
    lengths = torch.tensor([2])
    ranks = torch.zeros((4, 1), dtype=torch.int64)
    out = torch.empty(2, dtype=torch.int64)
    rows = 2
    if case == "out":
        out = out.int()
    elif case == "rows":
        rows = 3
    elif case == "rank_shape":
        ranks = ranks[:3]
    elif case == "boundaries":
        starts = starts[:1]
    with pytest.raises(ValueError):
        map_ckv_current_chunk_slots(
            starts,
            requests,
            lengths,
            ranks,
            padded_tokens=16,
            dcp_world_size=4,
            interleave=1,
            out=out,
            num_tokens=rows,
        )


@pytest.mark.parametrize("interleave", [1, 4])
def test_fp8_dsa_changed_chunk_matches_native_bytes_and_request_mapping(interleave):
    require_b12x()
    from b12x.attention._shared.mla.kv_cache import (
        concat_and_cache_fp8_ds_mla,
        map_ckv_current_chunk_slots,
    )

    device = torch.device("cuda")
    starts = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    requests = torch.tensor([0, 0, 0, 1, 1, -1], dtype=torch.int32, device=device)
    lengths = torch.tensor([11, 6], dtype=torch.int32, device=device)
    # Sliced rank rows exercise a non-contiguous request dimension extent.
    rank_storage = torch.zeros((4, 5), dtype=torch.int64, device=device)
    rank_storage[:, 1] = 16
    rank_starts = rank_storage[:, :2]
    slots = torch.full((6,), -2, dtype=torch.int64, device=device)
    cache = torch.full((4, 32, 656), 0xA5, dtype=torch.uint8, device=device)
    latent = torch.empty((6, 520), dtype=torch.bfloat16, device=device)[:, :512]
    rope = torch.empty((6, 72), dtype=torch.bfloat16, device=device)[:, :64]
    for seed in [17, 29]:
        torch.manual_seed(seed)
        latent.normal_()
        rope.normal_()
        map_ckv_current_chunk_slots(
            starts,
            requests,
            lengths,
            rank_starts,
            padded_tokens=32,
            dcp_world_size=4,
            interleave=interleave,
            out=slots,
            num_tokens=6,
        )
        expected_slots = []
        for req, position in [(0, 8), (0, 9), (0, 10), (1, 4), (1, 5)]:
            owner = (position // interleave) % 4
            local = position // (4 * interleave) * interleave + position % interleave
            expected_slots.append(owner * 32 + req * 16 + local)
        assert slots.cpu().tolist() == expected_slots + [-1]
        before = cache.clone()
        concat_and_cache_fp8_ds_mla(latent, rope, cache, slots)
        expected = _fp8_dsa_record_reference(latent[:5], rope[:5])
        flat = cache.view(-1, 656)
        assert torch.equal(flat[expected_slots].cpu(), expected)
        untouched = torch.ones(128, dtype=torch.bool, device=device)
        untouched[expected_slots] = False
        assert torch.equal(flat[untouched], before.view(-1, 656)[untouched])


def test_ckv_current_chunk_mapping_uses_64bit_rank_offsets():
    require_b12x()
    from b12x.attention._shared.mla.kv_cache import map_ckv_current_chunk_slots

    device = torch.device("cuda")
    offset = 2**31 + 64
    starts = torch.tensor([0, 2], device=device)
    reqs = torch.zeros(2, dtype=torch.int64, device=device)
    lengths = torch.tensor([2], device=device)
    ranks = torch.full((4, 1), offset, dtype=torch.int64, device=device)
    output = torch.empty(2, dtype=torch.int64, device=device)
    padded = offset + 128
    map_ckv_current_chunk_slots(
        starts,
        reqs,
        lengths,
        ranks,
        padded_tokens=padded,
        dcp_world_size=4,
        interleave=1,
        out=output,
        num_tokens=2,
    )
    assert output.cpu().tolist() == [offset, padded + offset]


def test_fp8_dsa_writer_addresses_records_beyond_signed_int32_bytes():
    require_b12x()
    from b12x.attention._shared.mla.kv_cache import concat_and_cache_fp8_ds_mla

    device = torch.device("cuda")
    block_size = 64
    block_stride = block_size * 656
    large_page = 2**31 // block_stride + 1
    # A mostly uninitialized pool places the live record beyond a 32-bit offset.
    cache = torch.empty(
        (large_page + 1, block_size, 656), dtype=torch.uint8, device=device
    )
    latent = torch.full((1, 512), 2.0, dtype=torch.bfloat16, device=device)
    rope = torch.full((1, 64), 3.0, dtype=torch.bfloat16, device=device)
    slots = torch.tensor([large_page * block_size], device=device)
    concat_and_cache_fp8_ds_mla(latent, rope, cache, slots)
    expected = _fp8_dsa_record_reference(latent, rope)
    assert torch.equal(cache[large_page, :1].cpu(), expected)


def test_fp8_dsa_chunk_live_rows_reuse_compiled_kernels(monkeypatch):
    require_b12x()
    from b12x.attention._shared.mla import kv_cache as writer

    device = torch.device("cuda")
    starts = torch.tensor([0, 5], dtype=torch.int32, device=device)
    reqs = torch.zeros(5, dtype=torch.int32, device=device)
    lengths = torch.tensor([5], dtype=torch.int32, device=device)
    ranks = torch.zeros((4, 1), dtype=torch.int32, device=device)
    slots = torch.full((5,), -1, dtype=torch.int64, device=device)
    latent = torch.ones((5, 512), dtype=torch.bfloat16, device=device)
    rope = torch.ones((5, 64), dtype=torch.bfloat16, device=device)
    cache = torch.zeros((4, 8, 656), dtype=torch.uint8, device=device)

    def run(rows):
        writer.map_ckv_current_chunk_slots(
            starts,
            reqs,
            lengths,
            ranks,
            padded_tokens=8,
            dcp_world_size=4,
            interleave=1,
            out=slots,
            num_tokens=rows,
        )
        writer.concat_and_cache_fp8_ds_mla(latent, rope, cache, slots, num_tokens=rows)

    run(1)

    def reject_compile(*args, **kwargs):
        raise AssertionError("live row count changed the compiled callable")

    monkeypatch.setattr(writer._map_ckv_chunk_slots_kernel, "compile", reject_compile)
    monkeypatch.setattr(
        writer._concat_and_cache_fp8_ds_mla_kernel, "compile", reject_compile
    )
    for rows in [2, 5, 1]:
        latent.fill_(rows)
        rope.fill_(rows + 1)
        run(rows)
        selected = slots[:rows].cpu().tolist()
        expected = _fp8_dsa_record_reference(latent[:rows], rope[:rows])
        assert torch.equal(cache.view(-1, 656)[selected].cpu(), expected)


def test_fp8_dsa_writer_skips_invalid_slots_and_zero_rows():
    require_b12x()
    from b12x.attention._shared.mla.kv_cache import concat_and_cache_fp8_ds_mla

    device = torch.device("cuda")
    latent = torch.ones((2, 512), dtype=torch.bfloat16, device=device)
    rope = torch.ones((2, 64), dtype=torch.bfloat16, device=device)
    cache = torch.full((1, 4, 656), 0xA5, dtype=torch.uint8, device=device)
    slots = torch.tensor([-1, 4], dtype=torch.int64, device=device)
    concat_and_cache_fp8_ds_mla(latent, rope, cache, slots)
    assert torch.all(cache == 0xA5)
    slots.zero_()
    concat_and_cache_fp8_ds_mla(latent, rope, cache, slots, num_tokens=0)
    assert torch.all(cache == 0xA5)


def test_fp8_dsa_writer_matches_vllm_native_record_bytes():
    require_b12x()
    ops = pytest.importorskip("vllm._custom_ops")
    from b12x.attention._shared.mla.kv_cache import concat_and_cache_fp8_ds_mla

    device = torch.device("cuda")
    latent = torch.zeros((6, 512), dtype=torch.bfloat16, device=device)
    rope = torch.empty((6, 64), dtype=torch.bfloat16, device=device)
    slots = torch.arange(6, dtype=torch.int64, device=device)
    candidate = torch.empty((1, 8, 656), dtype=torch.uint8, device=device)
    native = torch.empty_like(candidate)
    for seed, outer_scale in [(7, 0.25), (11, 4.0)]:
        torch.manual_seed(seed)
        latent.normal_()
        rope.normal_()
        latent[0].zero_()
        latent[1].fill_(torch.finfo(torch.bfloat16).tiny)
        latent[2].fill_(-448.0)
        # Group maxima pin scale to one; adjacent values exercise E4M3 ties.
        latent[3].fill_(448.0)
        latent[3, :8] = torch.tensor(
            [1.0, 1.0625, 1.125, 1.1875, -1.0, -1.0625, -1.125, -1.1875], device=device
        )
        scale = torch.tensor(outer_scale, dtype=torch.float32, device=device)
        concat_and_cache_fp8_ds_mla(latent, rope, candidate, slots)
        ops.concat_and_cache_mla(latent, rope, native, slots, "fp8_ds_mla", scale)
        assert torch.equal(candidate[0, :6], native[0, :6])


def _local_token_count(total, rank, dcp, interleave):
    cycles, remainder = divmod(total, dcp * interleave)
    return cycles * interleave + min(max(remainder - rank * interleave, 0), interleave)


def test_native_chunk_copy_rejects_cpu_and_invalid_capacity():
    from b12x.attention._shared.mla.kv_cache import (
        gather_ckv_current_chunk,
        gather_ckv_history,
        insert_ckv_current_chunk,
    )

    cache = torch.empty((2, 4, 656), dtype=torch.uint8)
    output = torch.empty((8, 656), dtype=torch.uint8)
    table = torch.tensor([[0, 1]], dtype=torch.int32)
    starts = torch.zeros((4, 1), dtype=torch.int32)
    lengths = torch.ones((4, 1), dtype=torch.int32)
    full = torch.tensor([4], dtype=torch.int32)
    queries = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="CUDA"):
        gather_ckv_history(
            cache,
            output,
            table,
            starts,
            lengths,
            full,
            queries,
            dcp_rank=0,
            dcp_world_size=4,
            interleave=1,
            num_reqs=1,
            padded_tokens=8,
        )
    with pytest.raises(ValueError, match="capacity"):
        gather_ckv_current_chunk(
            cache,
            output,
            table,
            full,
            queries,
            dcp_rank=0,
            dcp_world_size=4,
            interleave=1,
            num_reqs=1,
            current_capacity=9,
        )
    with pytest.raises(ValueError, match="CUDA"):
        gather_ckv_current_chunk(
            cache,
            output,
            table,
            full,
            queries,
            dcp_rank=0,
            dcp_world_size=4,
            interleave=1,
            num_reqs=1,
            current_capacity=8,
        )
    with pytest.raises(ValueError, match="capacity"):
        insert_ckv_current_chunk(
            output,
            torch.empty_like(output),
            starts,
            full,
            queries,
            dcp_world_size=4,
            interleave=1,
            num_reqs=1,
            current_capacity=3,
            padded_tokens=2,
        )


@pytest.mark.parametrize("interleave", [1, 4])
@pytest.mark.parametrize("record_bytes", [368, 656])
def test_native_history_and_chunk_exchange_match_full_gather(interleave, record_bytes):
    require_b12x()
    from b12x.attention._shared.mla.kv_cache import (
        gather_ckv_current_chunk,
        gather_ckv_history,
        insert_ckv_current_chunk,
    )

    device = torch.device("cuda")
    full_lengths = [11, 6]
    chunks = [3, 2]
    dcp, page_size, padded, capacity = 4, 4, 16, 8
    full = torch.tensor(full_lengths, dtype=torch.int32, device=device)
    queries = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    tables = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=device
    )
    lengths_cpu = [
        [_local_token_count(length, rank, dcp, interleave) for length in full_lengths]
        for rank in range(dcp)
    ]
    starts_cpu = [[0, lengths[0]] for lengths in lengths_cpu]
    # Strided rank metadata preserves builder-owned capacity beyond live requests.
    starts_storage = torch.zeros((dcp, 5), dtype=torch.int64, device=device)
    lens_storage = torch.zeros_like(starts_storage)
    starts = starts_storage[:, :2]
    lengths = lens_storage[:, :2]
    starts.copy_(torch.tensor(starts_cpu, device=device))
    lengths.copy_(torch.tensor(lengths_cpu, device=device))
    staging = [
        torch.empty((padded, record_bytes), dtype=torch.uint8, device=device)
        for _ in range(dcp)
    ]
    compact = [
        torch.empty((capacity, record_bytes), dtype=torch.uint8, device=device)
        for _ in range(dcp)
    ]
    for seed in [3, 19]:
        expected_ranks, expected_history, expected_current = [], [], []
        for rank in range(dcp):
            raw = (
                (
                    torch.arange(8 * page_size * record_bytes, dtype=torch.int64)
                    + seed
                    + rank * 31
                )
                % 251
                + 1
            ).to(torch.uint8)
            cpu_cache = raw.view(8, page_size, record_bytes)
            cache = cpu_cache.to(device)
            expected = torch.zeros((padded, record_bytes), dtype=torch.uint8)
            history = torch.zeros_like(expected)
            packed = []
            for req, count in enumerate(lengths_cpu[rank]):
                previous = _local_token_count(
                    full_lengths[req] - chunks[req], rank, dcp, interleave
                )
                for position in range(count):
                    record = cpu_cache[
                        req * 4 + position // page_size, position % page_size
                    ]
                    slot = starts_cpu[rank][req] + position
                    expected[slot] = record
                    if position < previous:
                        history[slot] = record
                    else:
                        packed.append(record)
            packed_expected = torch.zeros((capacity, record_bytes), dtype=torch.uint8)
            if packed:
                packed_expected[: len(packed)] = torch.stack(packed)
            gather_ckv_history(
                cache,
                staging[rank],
                tables,
                starts,
                lengths,
                full,
                queries,
                dcp_rank=rank,
                dcp_world_size=dcp,
                interleave=interleave,
                num_reqs=2,
                padded_tokens=padded,
            )
            assert torch.equal(staging[rank].cpu(), history)
            gather_ckv_current_chunk(
                cache,
                compact[rank],
                tables,
                full,
                queries,
                dcp_rank=rank,
                dcp_world_size=dcp,
                interleave=interleave,
                num_reqs=2,
                current_capacity=capacity,
            )
            assert torch.equal(compact[rank].cpu(), packed_expected)
            expected_ranks.append(expected)
            expected_history.append(history)
            expected_current.append(packed_expected)
        gathered = torch.cat(staging)
        packed_all = torch.cat(compact)
        insert_ckv_current_chunk(
            packed_all,
            gathered.view(-1, page_size, record_bytes),
            starts,
            full,
            queries,
            dcp_world_size=dcp,
            interleave=interleave,
            num_reqs=2,
            current_capacity=capacity,
            padded_tokens=padded,
        )
        assert torch.equal(gathered.cpu(), torch.cat(expected_ranks))


def test_native_history_gather_excludes_unwritten_chunk_pages():
    require_b12x()
    from b12x.attention._shared.mla.kv_cache import gather_ckv_history

    device = torch.device("cuda")
    cache = torch.full((1, 2, 656), 23, dtype=torch.uint8, device=device)
    output = torch.full((6, 656), 99, dtype=torch.uint8, device=device)
    # Rank 0 owns two produced history tokens and two future tokens. The future
    # tokens' table entry points outside the source allocation and must not load.
    table = torch.tensor([[0, 2**31]], dtype=torch.int64, device=device)
    starts = torch.zeros((4, 1), dtype=torch.int64, device=device)
    lengths = torch.full((4, 1), 4, dtype=torch.int64, device=device)
    full = torch.tensor([16], dtype=torch.int64, device=device)
    queries = torch.tensor([0, 8], dtype=torch.int64, device=device)
    gather_ckv_history(
        cache,
        output,
        table,
        starts,
        lengths,
        full,
        queries,
        dcp_rank=0,
        dcp_world_size=4,
        interleave=1,
        num_reqs=1,
        padded_tokens=6,
    )
    assert torch.all(output[:2] == 23)
    assert torch.all(output[2:] == 0)


def test_native_history_and_chunk_gather_address_large_recycled_pages():
    require_b12x()
    from b12x.attention._shared.mla.kv_cache import (
        gather_ckv_current_chunk,
        gather_ckv_history,
    )

    device = torch.device("cuda")
    record_bytes = 656
    page = 2**31 // record_bytes + 1
    cache = torch.empty((page + 2, 1, record_bytes), dtype=torch.uint8, device=device)
    cache[page].fill_(19)
    cache[page + 1].fill_(71)
    table = torch.tensor([[page, page + 1]], dtype=torch.int64, device=device)
    starts = torch.zeros((4, 1), dtype=torch.int64, device=device)
    lengths = torch.full((4, 1), 2, dtype=torch.int64, device=device)
    full = torch.tensor([8], dtype=torch.int64, device=device)
    queries = torch.tensor([0, 4], dtype=torch.int64, device=device)
    history = torch.empty((2, record_bytes), dtype=torch.uint8, device=device)
    current = torch.empty_like(history)
    gather_ckv_history(
        cache,
        history,
        table,
        starts,
        lengths,
        full,
        queries,
        dcp_rank=0,
        dcp_world_size=4,
        interleave=1,
        num_reqs=1,
        padded_tokens=2,
    )
    gather_ckv_current_chunk(
        cache,
        current,
        table,
        full,
        queries,
        dcp_rank=0,
        dcp_world_size=4,
        interleave=1,
        num_reqs=1,
        current_capacity=2,
    )
    assert torch.all(history[0] == 19)
    assert torch.all(history[1] == 0)
    assert torch.all(current[0] == 71)
    assert torch.all(current[1] == 0)


def test_native_copy_empty_requests_reuse_precompiled_capacity(monkeypatch):
    require_b12x()
    from b12x.attention._shared.mla import kv_cache as writer

    device = torch.device("cuda")
    cache = torch.ones((2, 4, 656), dtype=torch.uint8, device=device)
    history = torch.empty((4, 656), dtype=torch.uint8, device=device)
    current = torch.empty_like(history)
    full_output = torch.full((16, 656), 29, dtype=torch.uint8, device=device)
    gathered = torch.ones_like(full_output)
    table = torch.zeros((1, 2), dtype=torch.int32, device=device)
    starts = torch.zeros((4, 1), dtype=torch.int32, device=device)
    lengths = torch.ones_like(starts)
    full = torch.tensor([4], dtype=torch.int32, device=device)
    queries = torch.tensor([0, 4], dtype=torch.int32, device=device)

    def run(requests):
        writer.gather_ckv_history(
            cache,
            history,
            table,
            starts,
            lengths,
            full,
            queries,
            dcp_rank=0,
            dcp_world_size=4,
            interleave=1,
            num_reqs=requests,
            padded_tokens=4,
        )
        writer.gather_ckv_current_chunk(
            cache,
            current,
            table,
            full,
            queries,
            dcp_rank=0,
            dcp_world_size=4,
            interleave=1,
            num_reqs=requests,
            current_capacity=4,
        )
        writer.insert_ckv_current_chunk(
            gathered,
            full_output,
            starts,
            full,
            queries,
            dcp_world_size=4,
            interleave=1,
            num_reqs=requests,
            current_capacity=4,
            padded_tokens=4,
        )

    run(1)

    def reject_compile(*args, **kwargs):
        raise AssertionError("live request count changed native-copy compilation")

    for kernel in [
        writer._gather_ckv_history_kernel,
        writer._gather_ckv_current_chunk_kernel,
        writer._insert_ckv_current_chunk_kernel,
    ]:
        monkeypatch.setattr(kernel, "compile", reject_compile)
    full_output.fill_(29)
    run(0)
    assert torch.all(history == 0)
    assert torch.all(current == 0)
    assert torch.all(full_output == 29)
