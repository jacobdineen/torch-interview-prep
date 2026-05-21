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
from p18c_bce_with_logits2 import *

def test_p18c_bce_with_logits2():
    torch.manual_seed(0)
    logits = torch.randn(10, 3) * 4
    targets = (torch.rand(10, 3) > 0.5).float()
    for red in ('mean', 'sum', 'none'):
        pass
    pass
    with step('bce_with_logits is finite at extreme logits'):
        big = torch.tensor([[-10000.0, 0.0, 10000.0]])
        tgt = torch.tensor([[0.0, 1.0, 1.0]])
        out = bce_with_logits(big, tgt)
        assert torch.isfinite(out).all()
