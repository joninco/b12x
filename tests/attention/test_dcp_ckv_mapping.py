"""Stable gathered-KV slot mapping: CPU oracle/contracts and GPU row scans.

Inputs are request IDs, selected global IDs and rank-local request geometry.
The oracle checks ordered compaction, not only set membership. GPU cases remain
separate from CPU validation and include pitched storage and changing inputs.
"""

import pytest
import torch

from b12x.attention._shared.mla.dcp_ckv_mapping import map_global_topk_to_gathered_ckv
from tests._reference.helpers import require_b12x


def reference(req_ids, tokens, starts, lengths, *, dcp, interleave, padded):
    result = torch.full_like(tokens, -1, device="cpu")
    counts = torch.zeros(tokens.shape[0], dtype=torch.int32)
    for row, request in enumerate(req_ids.tolist()):
        if not 0 <= request < starts.shape[1]:
            continue
        survivors = []
        for token in tokens[row].tolist():
            if token < 0:
                continue
            owner = token // interleave % dcp
            local = token // (dcp * interleave) * interleave + token % interleave
            start, length = int(starts[owner, request]), int(lengths[owner, request])
            if local < length and start >= 0 and start + local < padded:
                survivors.append(owner * padded + start + local)
        counts[row] = len(survivors)
        result[row, : len(survivors)] = torch.tensor(survivors, dtype=torch.int32)
    return result, counts


def test_reference_preserves_holes_duplicates_and_cross_rank_input_order():
    ids = torch.tensor([0, -1, 1], dtype=torch.int32)
    tokens = torch.tensor([[4, -1, 1, 0, 4, 12, 2, 3]] * 3, dtype=torch.int32)
    starts = torch.tensor([[1], [3], [5], [7]], dtype=torch.int32)
    lengths = torch.tensor([[2], [1], [1], [0]], dtype=torch.int32)
    mapped, counts = reference(
        ids, tokens, starts, lengths, dcp=4, interleave=1, padded=16
    )
    assert mapped[0].tolist() == [2, 19, 1, 2, 37, -1, -1, -1]
    assert counts.tolist() == [5, 0, 0]
    assert torch.all(mapped[1:] == -1)


def _args(rows=3, width=2048, device="cpu"):
    request_storage = torch.zeros(rows * 2, dtype=torch.int32, device=device)
    ids = request_storage[::2]
    token_storage = torch.empty((rows, width * 2 + 8), dtype=torch.int32, device=device)
    tokens = token_storage[:, : width * 2 : 2]
    tokens.copy_(torch.arange(width, dtype=torch.int32, device=device).expand(rows, -1))
    starts = torch.zeros((8, 6), dtype=torch.int32, device=device)[::2, ::2]
    lengths = torch.full((8, 6), 256, dtype=torch.int32, device=device)[::2, ::2]
    output_storage = torch.full(
        (rows + 2, width * 2 + 8), 91, dtype=torch.int32, device=device
    )
    out = output_storage[1 : rows + 1, : width * 2 : 2]
    count_storage = torch.full((rows * 2,), 73, dtype=torch.int32, device=device)
    counts = count_storage[::2]
    return [ids, tokens, starts, lengths, out, counts], (output_storage, count_storage)


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "shape",
        "width",
        "req_shape",
        "rank_shape",
        "capacity",
        "overlap",
        "device",
    ],
)
def test_mapping_rejects_invalid_cpu_contract(case):
    args, _ = _args()
    padded = 512
    if case == "dtype":
        args[0] = args[0].long()
    elif case == "shape":
        args[4] = args[4][:, :-1]
    elif case == "width":
        args[1] = torch.zeros((3, 4097), dtype=torch.int32)
    elif case == "req_shape":
        args[0] = args[0][:1]
    elif case == "rank_shape":
        args[2] = args[2][:3]
    elif case == "capacity":
        padded = 2**30
    elif case == "overlap":
        args[4] = args[1]
    with pytest.raises((ValueError, TypeError)):
        map_global_topk_to_gathered_ckv(
            *args, dcp_size=4, cp_kv_cache_interleave_size=1, padded_rank_tokens=padded
        )


@pytest.mark.parametrize("width", [1, 127, 128, 129, 2047, 2048, 2051, 4096])
@pytest.mark.parametrize("interleave", [1, 4])
def test_mapping_stable_full_row_scan_matches_oracle(width, interleave):
    require_b12x()
    args, (output_storage, count_storage) = _args(rows=5, width=width, device="cuda")
    ids, tokens, starts, lengths, out, counts = args
    ids.copy_(torch.tensor([0, 1, 2, -1, 3], device="cuda"))
    starts[:, 1] = torch.tensor([11, 21, 31, 41], device="cuda")
    lengths[:, 2] = 0
    for seed in [3, 17]:
        torch.manual_seed(seed)
        tokens.random_(-5, 1300)
        tokens[:, ::7] = -1
        expected, expected_counts = reference(
            *(t.cpu() for t in args[:4]), dcp=4, interleave=interleave, padded=512
        )
        for _ in range(8):
            out.fill_(99)
            counts.fill_(99)
            map_global_topk_to_gathered_ckv(
                *args,
                dcp_size=4,
                cp_kv_cache_interleave_size=interleave,
                padded_rank_tokens=512,
            )
            assert torch.equal(out.cpu(), expected)
            assert torch.equal(counts.cpu(), expected_counts)
        assert torch.all(output_storage[0] == 91)
        assert torch.all(output_storage[-1] == 91)
        assert torch.all(output_storage[1:-1, 1::2] == 91)
        assert torch.all(count_storage[1::2] == 73)
    tokens.fill_(-1)
    map_global_topk_to_gathered_ckv(
        *args,
        dcp_size=4,
        cp_kv_cache_interleave_size=interleave,
        padded_rank_tokens=512,
    )
    assert torch.all(out == -1)
    assert torch.all(counts == 0)


def test_mapping_live_rows_and_rank_span_reuse_compilation(monkeypatch):
    require_b12x()
    from b12x.attention._shared.mla.dcp_ckv_mapping import (
        _map_global_topk_to_gathered_ckv_kernel,
    )

    args, _ = _args(rows=3, device="cuda")
    map_global_topk_to_gathered_ckv(
        *args, dcp_size=4, cp_kv_cache_interleave_size=1, padded_rank_tokens=512
    )

    def reject_compile(*a, **kw):
        raise AssertionError("live rows or padded context entered CKV compile key")

    monkeypatch.setattr(
        _map_global_topk_to_gathered_ckv_kernel, "compile", reject_compile
    )
    for rows, padded in [(1, 1), (2, 16), (3, 257)]:
        sliced = [
            args[0][:rows],
            args[1][:rows],
            args[2],
            args[3],
            args[4][:rows],
            args[5][:rows],
        ]
        map_global_topk_to_gathered_ckv(
            *sliced,
            dcp_size=4,
            cp_kv_cache_interleave_size=1,
            padded_rank_tokens=padded,
        )
        expected, counts = reference(
            *(t.cpu() for t in sliced[:4]), dcp=4, interleave=1, padded=padded
        )
        assert torch.equal(sliced[4].cpu(), expected)
        assert torch.equal(sliced[5].cpu(), counts)


def test_mapping_empty_rows_and_empty_request_table():
    require_b12x()
    args, _ = _args(rows=0, device="cuda")
    map_global_topk_to_gathered_ckv(
        *args, dcp_size=4, cp_kv_cache_interleave_size=1, padded_rank_tokens=0
    )
    args, _ = _args(rows=1, device="cuda")
    args[2] = args[2][:, :0]
    args[3] = args[3][:, :0]
    map_global_topk_to_gathered_ckv(
        *args, dcp_size=4, cp_kv_cache_interleave_size=1, padded_rank_tokens=0
    )
    assert torch.all(args[4] == -1)
    assert args[5].item() == 0


def test_mapping_large_output_row_byte_stride():
    require_b12x()
    args, _ = _args(rows=2, width=2048, device="cuda")
    stride = 2**31 // 4 + 16
    storage = torch.empty(stride + 2048, dtype=torch.int32, device="cuda")
    args[4] = storage.as_strided((2, 2048), (stride, 1))
    map_global_topk_to_gathered_ckv(
        *args, dcp_size=4, cp_kv_cache_interleave_size=1, padded_rank_tokens=512
    )
    expected, counts = reference(
        *(t.cpu() for t in args[:4]), dcp=4, interleave=1, padded=512
    )
    assert torch.equal(args[4].cpu(), expected)
    assert torch.equal(args[5].cpu(), counts)
