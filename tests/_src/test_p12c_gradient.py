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
from p12c_gradient import *

def test_p12c_gradient():
    torch.manual_seed(0)
    pass
    pass
    with step('gradient equals sigmoid(x)'):
        x3 = torch.randn(4, requires_grad=True)
        out = stable_softplus(x3).sum()
        out.backward()
        assert torch.allclose(x3.grad, torch.sigmoid(x3), atol=1e-05)
