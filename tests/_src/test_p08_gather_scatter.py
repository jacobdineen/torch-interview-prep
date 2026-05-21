import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import torch
from p08_gather_scatter import *

def test_p08_gather_scatter():

    with step("gather_per_row picks one column per row"):
        x = torch.tensor([[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]])
        idx = torch.tensor([2, 0, 1])
        assert torch.equal(gather_per_row(x, idx), torch.tensor([30., 40., 80.]))
    with step("one_hot makes a (N, C) float matrix"):
        labels = torch.tensor([0, 2, 1, 2])
        oh = one_hot(labels, num_classes=3)
        assert oh.shape == (4, 3)
        assert oh.dtype.is_floating_point
        assert torch.equal(
            oh,
            torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 0., 1.]]),
        )
    with step("scatter_sum_rows sums values by row index (segment sum)"):
        values = torch.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.]])
        index = torch.tensor([0, 1, 0, 2])
        out = scatter_sum_rows(values, index, num_rows=3)
        assert out.shape == (3, 2)
        assert torch.allclose(out, torch.tensor([[4., 4.], [2., 2.], [4., 4.]]))

