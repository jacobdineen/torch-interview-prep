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
from p17d_huber_loss import *

def test_p17d_huber_loss():
    torch.manual_seed(0)
    p = torch.randn(8, 5)
    t = torch.randn(8, 5)
    pass
    pass
    pass
    with step('huber_loss with delta=0.7 matches F.huber_loss'):
        expected = torch.nn.functional.huber_loss(p, t, delta=0.7, reduction='mean')
        assert torch.allclose(huber_loss(p, t, delta=0.7), expected, atol=1e-06)
    pass
