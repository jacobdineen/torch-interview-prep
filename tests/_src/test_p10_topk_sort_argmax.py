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
from p10_topk_sort_argmax import *

def test_p10_topk_sort_argmax():

    x = torch.tensor([[1., 5., 3., 2., 4.],
                      [9., 0., 7., 8., 6.]])
    with step("top_k_per_row returns top-k values and indices, sorted desc"):
        vals, idx = top_k_per_row(x, k=3)
        assert vals.shape == (2, 3) and idx.shape == (2, 3)
        assert torch.equal(vals[0], torch.tensor([5., 4., 3.]))
        assert torch.equal(idx[0], torch.tensor([1, 4, 2]))
    with step("sort_then_take_indices returns k smallest by index (ascending order of value)"):
        bottom2 = sort_then_take_indices(x, k=2)
        assert bottom2.shape == (2, 2)
        assert torch.equal(bottom2[0], torch.tensor([0, 3]))
        assert torch.equal(bottom2[1], torch.tensor([1, 4]))
    with step("argmax_2d returns the (row, col) of the global max"):
        grid = torch.tensor([[0., 1., 2.], [3., 9., 4.], [5., 6., 7.]])
        rc = argmax_2d(grid)
        assert torch.equal(rc, torch.tensor([1, 1]))
    with step("kth_largest picks the k-th largest (1-indexed) in a flat tensor"):
        flat = torch.tensor([3., 1., 4., 1., 5., 9., 2., 6.])
        assert kth_largest(flat, 1) == 9
        assert kth_largest(flat, 3) == 5
        assert kth_largest(flat, 8) == 1

