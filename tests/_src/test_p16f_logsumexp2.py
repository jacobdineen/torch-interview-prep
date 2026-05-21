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
from p16f_logsumexp2 import *

def test_p16f_logsumexp2():
    torch.manual_seed(0)
    x = torch.randn(4, 5)
    pass
    pass
    pass
    pass
    pass
    with step('logsumexp stable at logits of order 1e4'):
        big = torch.tensor([[10000.0, 10000.0 + 1.0, 10000.0 - 1.0]])
        lse = logsumexp(big, dim=1)
        assert torch.isfinite(lse).all()
