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
from p19c_cross_entropy2 import *

def test_p19c_cross_entropy2():
    torch.manual_seed(0)
    logits = torch.randn(7, 5) * 3
    targets = torch.randint(0, 5, (7,))
    for red in ('mean', 'sum', 'none'):
        pass
    pass
    with step('cross_entropy stable at logits of order 1e4'):
        big = torch.tensor([[10000.0, 10000.0 + 1.0, 10000.0 - 1.0]])
        loss = cross_entropy(big, torch.tensor([1]))
        assert torch.isfinite(loss).all()
