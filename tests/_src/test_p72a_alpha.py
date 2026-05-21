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
from p72a_alpha import *

def test_p72a_alpha():
    torch.manual_seed(0)
    N, C = (8, 5)
    s = torch.randn(N, C, requires_grad=True)
    t = torch.randn(N, C)
    y = torch.randint(0, C, (N,))
    with step('alpha=1 equals plain CE'):
        loss = kd_loss(s, t, y, T=2.0, alpha=1.0)
        assert torch.allclose(loss, F.cross_entropy(s, y), atol=1e-06)
    pass
    pass
    pass
