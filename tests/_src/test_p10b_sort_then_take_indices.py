import contextlib


@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError with the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import contextlib
import torch
from p10b_sort_then_take_indices import *

def test_p10b_sort_then_take_indices():
    x = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0], [9.0, 0.0, 7.0, 8.0, 6.0]])
    pass
    with step('sort_then_take_indices returns k smallest by index (ascending order of value)'):
        bottom2 = sort_then_take_indices(x, k=2)
        assert bottom2.shape == (2, 2)
        assert torch.equal(bottom2[0], torch.tensor([0, 3]))
        assert torch.equal(bottom2[1], torch.tensor([1, 4]))
    pass
    pass
