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
from p11c_second_derivative import *

def test_p11c_second_derivative():
    torch.manual_seed(0)
    pass
    pass
    with step('second_derivative of sum(x^4) equals 12*x^2'):
        x2 = torch.tensor([1.0, 2.0, 3.0])
        h = second_derivative(x2)
        assert torch.allclose(h, 12 * x2 ** 2)
