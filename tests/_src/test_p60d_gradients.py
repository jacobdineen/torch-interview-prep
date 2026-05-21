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
import torch.nn.functional as F
from p60d_gradients import *

def test_p60d_gradients():
    torch.manual_seed(0)
    D, FF = (16, 32)
    blk = SwiGLU(D, FF)
    x = torch.randn(2, 7, D)
    out = blk(x)
    pass
    pass
    pass
    with step('gradients are finite for every parameter'):
        out.sum().backward()
        for name, p in blk.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
