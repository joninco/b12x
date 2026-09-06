"""Tests for the top-k selection sort (``b12x::topk_sort_convert``).

Contract under test: each row of an int32 ``[rows, topk]`` selection holding
logical KV positions (``-1`` unused) is rewritten in place ascending by
position and converted to physical cache slots through the row's block table;
the tail is ``-1``; duplicates collapse; the result is bitwise repeatable, a
function of the selected set alone; the kernel allocates nothing and can be
captured on a side stream inside a CUDA graph after ``precompile``.
"""

from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


def _api():
    from b12x.attention import topk_sort

    topk_sort.sort_convert  # noqa: B018  (registers the op)
    return topk_sort


def _case(
    rows: int,
    topk: int,
    seq_lens: list[int],
    block_size: int,
    device,
    *,
    seed: int = 0,
    fill: float = 1.0,
):
    """Random distinct positions below each row's length, written in a
    shuffled (arrival) order; the block table maps pages to random slots."""
    g = torch.Generator().manual_seed(seed)
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    width = max((max(seq_lens) + block_size - 1) // block_size, 1)
    pages = max(4096, rows * width)
    block_table = torch.randperm(pages, generator=g)[: rows * width].view(rows, width)
    for row in range(rows):
        count = min(topk, int(seq_lens[row] * fill))
        positions = torch.randperm(seq_lens[row], generator=g)[:count]
        indices[row, :count] = positions.to(torch.int32)
    return (
        indices.to(device),
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
        block_table.to(torch.int32).to(device),
    )


def test_bitmap_words_limits():
    api = _api()
    assert api.bitmap_words(32) == 1
    assert api.bitmap_words(33) == 2
    assert api.bitmap_words(32768) == 1024
    with pytest.raises(ValueError):
        api.bitmap_words(0)
    with pytest.raises(ValueError):
        api.bitmap_words(api.MAX_BITMAP_WORDS * 32 + 1)


def test_registry_lists_op():
    import b12x
    from b12x import attention

    assert "attention.topk_sort" in b12x._OPS
    assert "topk_sort" in attention._OP_MODULES


def test_reference_semantics_on_cpu():
    api = _api()
    indices = torch.tensor(
        [[70, 3, -1, 3, 65, 200], [1, 0, 2, -1, -1, -1]], dtype=torch.int32
    )
    seq_lens = torch.tensor([100, 3], dtype=torch.int32)
    block_table = torch.tensor([[5, 9], [7, 0]], dtype=torch.int32)
    out = api.sort_convert_reference(indices, seq_lens, block_table, 64, 4096)
    # Row 0: positions 3, 65, 70 (200 is beyond the 128-position bitmap limit
    # of a 100-token row; the duplicate 3 collapses): page 5 slot 3, page 9
    # slots 1 and 6.
    assert out[0].tolist() == [5 * 64 + 3, 9 * 64 + 1, 9 * 64 + 6, -1, -1, -1]
    assert out[1].tolist() == [7 * 64 + 0, 7 * 64 + 1, 7 * 64 + 2, -1, -1, -1]


@cuda_required
@pytest.mark.parametrize(
    "rows,topk,seq_lens,block_size,max_positions",
    [
        (4, 2048, [1000, 1500, 2047, 2048], 64, 32768),
        (8, 2048, [30000, 20000, 4096, 4097, 33, 32, 1, 8192], 64, 32768),
        (16, 2048, [12345] * 16, 64, 131072),
        (3, 512, [600, 511, 512], 16, 4096),
        (2, 2048, [40000, 65536], 64, 65536),
        # The GLM-5.3 launch geometry: a 524,288-token model length (16,384
        # bitmap words, 64 KiB of shared memory) with rows from 30k to the
        # last position.
        (4, 2048, [30000, 131072, 200000, 524287], 64, 524288),
    ],
)
def test_matches_reference(rows, topk, seq_lens, block_size, max_positions):
    api = _api()
    device = torch.device("cuda")
    indices, lens, table = _case(rows, topk, seq_lens, block_size, device)
    expected = api.sort_convert_reference(
        indices.cpu(), lens.cpu(), table.cpu(), block_size, max_positions
    )
    api.sort_convert(indices, lens, table, block_size, max_positions)
    torch.cuda.synchronize(device)
    assert torch.equal(indices.cpu(), expected)


@cuda_required
def test_sparse_and_dense_rows_and_duplicates():
    api = _api()
    device = torch.device("cuda")
    # Dense: every position of a 2048-token row selected (fill 1.0 with
    # topk == seq_len) and a nearly empty row; duplicates in a third row.
    indices, lens, table = _case(3, 2048, [2048, 5, 3000], 64, device, fill=1.0)
    indices[2, :10] = torch.tensor(
        [9, 9, 9, 2999, 2999, 0, 0, 1, 1, 2], dtype=torch.int32, device=device
    )
    indices[2, 10:] = -1
    expected = api.sort_convert_reference(
        indices.cpu(), lens.cpu(), table.cpu(), 64, 32768
    )
    api.sort_convert(indices, lens, table, 64, 32768)
    torch.cuda.synchronize(device)
    assert torch.equal(indices.cpu(), expected)
    assert indices[2, 5:].eq(-1).all()


@cuda_required
def test_order_depends_only_on_the_selected_set():
    """Two arrival orders of the same set give bitwise identical rows."""
    api = _api()
    device = torch.device("cuda")
    indices, lens, table = _case(4, 2048, [30000] * 4, 64, device)
    g = torch.Generator().manual_seed(3)
    shuffled = indices.clone()
    for row in range(4):
        perm = torch.randperm(2048, generator=g).to(device)
        shuffled[row] = indices[row][perm]
    api.sort_convert(indices, lens, table, 64, 32768)
    api.sort_convert(shuffled, lens, table, 64, 32768)
    torch.cuda.synchronize(device)
    assert torch.equal(indices, shuffled)
    first = indices.clone()
    for _ in range(20):
        again = shuffled.clone()
        # Already physical slots would be re-sorted as positions; rebuild the
        # logical input instead from the reference case each time.
        again, _, _ = _case(4, 2048, [30000] * 4, 64, device)
        api.sort_convert(again, lens, table, 64, 32768)
        assert torch.equal(again, first)


@cuda_required
def test_expanded_block_table_and_strided_rows():
    """A single-request prefill chunk uses one block-table row expanded over
    the chunk rows (stride 0) and a row-strided index view."""
    api = _api()
    device = torch.device("cuda")
    indices, lens, table = _case(6, 2048, [9000] * 6, 64, device)
    wide = torch.full((6, 4096), -1, dtype=torch.int32, device=device)
    wide[:, :2048] = indices
    view = wide[:, :2048]
    expanded = table[:1].expand(6, -1)
    expected = api.sort_convert_reference(
        view.cpu(), lens.cpu(), expanded.cpu(), 64, 32768
    )
    api.sort_convert(view, lens, expanded, 64, 32768)
    torch.cuda.synchronize(device)
    assert torch.equal(view.cpu(), expected)
    assert wide[:, 2048:].eq(-1).all()


@cuda_required
def test_precompile_then_capture_on_side_stream():
    import b12x

    api = _api()
    device = torch.device("cuda")
    api.precompile(32768, device)
    indices, lens, table = _case(4, 2048, [20000] * 4, 64, device)
    logical = indices.clone()
    expected = api.sort_convert_reference(
        indices.cpu(), lens.cpu(), table.cpu(), 64, 32768
    )
    main = torch.cuda.Stream(device)
    side = torch.cuda.Stream(device)
    graph = torch.cuda.CUDAGraph()
    b12x.freeze_kernel_resolution("topk sort capture test")
    try:
        with torch.cuda.stream(main):
            torch.cuda.synchronize(device)
            with torch.cuda.graph(graph, stream=main):
                side.wait_stream(main)
                with torch.cuda.stream(side):
                    api.sort_convert(indices, lens, table, 64, 32768)
                main.wait_stream(side)
        torch.cuda.synchronize(device)
        allocated = torch.cuda.memory_allocated(device)
        for _ in range(5):
            indices.copy_(logical)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.equal(indices.cpu(), expected)
        assert torch.cuda.memory_allocated(device) == allocated
    finally:
        b12x.unfreeze_kernel_resolution()


@cuda_required
def test_rejects_out_of_contract_inputs():
    api = _api()
    device = torch.device("cuda")
    indices, lens, table = _case(2, 512, [100, 100], 64, device)
    assert api.supports(indices, lens, table, 64, 4096)
    assert not api.supports(indices, lens, table, 48, 4096)
    assert not api.supports(indices.to(torch.int64), lens, table, 64, 4096)
    assert not api.supports(indices, lens[:1], table, 64, 4096)
    assert not api.supports(indices, lens, table.to(torch.int64), 64, 4096)
    assert not api.supports(indices.t(), lens, table, 64, 4096)
    with pytest.raises(ValueError):
        api.sort_convert(indices, lens, table, 48, 4096)
    empty = indices[:0]
    api.sort_convert(empty, lens, table, 64, 4096)  # no rows: no launch
