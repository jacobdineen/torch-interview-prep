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
from p74f_hard import *

def test_p74f_hard():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    samples = gumbel_noise((100000,), generator=g)
    pass
    pass
    pass
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    pass
    pass
    with step('hard=True returns one-hot but keeps straight-through grad'):
        logits_r = torch.randn(5, 8, requires_grad=True)
        g4 = torch.Generator().manual_seed(0)
        y_hard = gumbel_softmax(logits_r, tau=1.0, hard=True, generator=g4)
        assert y_hard.shape == logits_r.shape
        assert torch.allclose(y_hard.sum(dim=-1), torch.ones(5))
        assert (y_hard == 0).logical_or(y_hard == 1).all()
        y_hard.sum().backward()
        assert logits_r.grad is not None and torch.isfinite(logits_r.grad).all()
