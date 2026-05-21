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
from p11a_grad_of_sum_of_squares import *

def test_p11a_grad_of_sum_of_squares():
    torch.manual_seed(0)
    with step('grad_of_sum_of_squares matches 2*x'):
        x = torch.tensor([1.0, -2.0, 3.0])
        g = grad_of_sum_of_squares(x)
        assert torch.allclose(g, 2 * x)
    pass
    pass
