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
from p10c_argmax_2d import *

def test_p10c_argmax_2d():
    x = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0], [9.0, 0.0, 7.0, 8.0, 6.0]])
    pass
    pass
    with step('argmax_2d returns the (row, col) of the global max'):
        grid = torch.tensor([[0.0, 1.0, 2.0], [3.0, 9.0, 4.0], [5.0, 6.0, 7.0]])
        rc = argmax_2d(grid)
        assert torch.equal(rc, torch.tensor([1, 1]))
    pass
