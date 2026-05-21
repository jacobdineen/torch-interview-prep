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
import torch.nn as nn
from p13b_stop_gradient_at import *

def test_p13b_stop_gradient_at():
    torch.manual_seed(0)
    pass
    with step('stop_gradient_at: forward equals x; grads blocked above threshold'):
        x = torch.tensor([0.5, 1.0, 2.0, 3.0], requires_grad=True)
        y = stop_gradient_at(x, threshold=1.5)
        assert torch.allclose(y, x.detach())
        s = y.sum()
        s.backward()
        assert torch.equal(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    pass
