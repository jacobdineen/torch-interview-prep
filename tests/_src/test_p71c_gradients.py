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
import math
import torch
import torch.nn.functional as F
from p71c_gradients import *

def test_p71c_gradients():
    torch.manual_seed(0)
    N, D = (64, 32)
    a = torch.randn(N, D)
    pass
    pass
    with step('gradients flow through both views'):
        a_g = torch.randn(N, D, requires_grad=True)
        b_g = torch.randn(N, D, requires_grad=True)
        l = info_nce_loss(a_g, b_g, tau=0.1)
        l.backward()
        assert a_g.grad is not None and torch.isfinite(a_g.grad).all()
        assert b_g.grad is not None and torch.isfinite(b_g.grad).all()
    pass
