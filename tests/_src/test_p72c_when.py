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
from p72c_when import *

def test_p72c_when():
    torch.manual_seed(0)
    N, C = (8, 5)
    s = torch.randn(N, C, requires_grad=True)
    t = torch.randn(N, C)
    y = torch.randint(0, C, (N,))
    pass
    pass
    with step('when teacher == student, KL term is 0; loss = alpha * CE'):
        loss = kd_loss(s, s.detach(), y, T=3.0, alpha=0.5)
        assert torch.allclose(loss, 0.5 * F.cross_entropy(s, y), atol=1e-05)
    pass
