"""Validate eager prefill LSE copy/masking and caller-owned output lifetime."""

import pytest
import torch

from b12x.comm.prefill import prepare_prefill_lse
from tests._reference.helpers import require_b12x


@pytest.mark.parametrize("case", ["dtype", "lengths", "rows", "overlap", "device"])
def test_prepare_prefill_lse_rejects_invalid_cpu_contract(case):
    source = torch.empty((3, 4), dtype=torch.float32)
    output = torch.empty_like(source)
    lengths = torch.ones(3, dtype=torch.int32)
    rows = 3
    if case == "dtype":
        source = source.double()
    elif case == "lengths":
        lengths = lengths.long()
    elif case == "rows":
        rows = 4
    elif case == "overlap":
        output = source
    with pytest.raises(ValueError):
        prepare_prefill_lse(source, lengths, output, num_rows=rows)


@pytest.mark.parametrize("layout", ["pitched", "transposed", "strided"])
def test_prepare_prefill_lse_masks_invalid_partials_and_preserves_canaries(layout):
    require_b12x()
    device = torch.device("cuda")
    if layout == "pitched":
        source = torch.empty((5, 12), device=device)[:, :4]
    elif layout == "transposed":
        source = torch.empty((4, 5), device=device).T
    else:
        source = torch.empty((10, 12), device=device)[::2, ::3]
    source.copy_(
        torch.tensor(
            [
                [1.0, float("nan"), float("inf"), -float("inf")],
                [2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0],
                [-2.0, 0.0, 2.0, -float("inf")],
                [9.0, 9.0, 9.0, 9.0],
            ],
            device=device,
        )
    )
    lengths = torch.tensor([1, 0, -2, 4, 5], dtype=torch.int32, device=device)
    storage = torch.full((7, 4), 713.0, device=device)
    output = storage[1:6]
    prepare_prefill_lse(source, lengths, output, num_rows=4)
    expected = torch.tensor(
        [
            [1.0, -float("inf"), -float("inf"), -float("inf")],
            [-float("inf")] * 4,
            [-float("inf")] * 4,
            [-2.0, 0.0, 2.0, -float("inf")],
        ],
        device=device,
    )
    torch.testing.assert_close(output[:4], expected)
    source.fill_(999.0)
    torch.testing.assert_close(output[:4], expected)
    assert torch.all(storage[0] == 713.0)
    assert torch.all(storage[5:] == 713.0)


def test_prepare_prefill_lse_live_rows_reuse_compiled_kernel(monkeypatch):
    require_b12x()
    from b12x.comm.prefill import _prepare_prefill_lse_kernel

    device = torch.device("cuda")
    source = torch.ones((5, 4), device=device)
    output = torch.empty_like(source)
    lengths = torch.ones(5, dtype=torch.int32, device=device)
    prepare_prefill_lse(source, lengths, output, num_rows=1)

    def reject_compile(*args, **kwargs):
        raise AssertionError("live rows changed prefill LSE compiled callable")

    monkeypatch.setattr(_prepare_prefill_lse_kernel, "compile", reject_compile)
    for rows in [2, 5, 1]:
        source.fill_(rows)
        prepare_prefill_lse(source, lengths, output, num_rows=rows)
        assert torch.all(output[:rows] == rows)
