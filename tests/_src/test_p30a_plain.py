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
from p30a_plain import *

def test_p30a_plain():
    torch.manual_seed(0)
    target = torch.tensor([1.0, -2.0, 3.0])
    x = torch.zeros(3, requires_grad=True)
    with step('plain SGD converges to target on a quadratic'):
        opt = MySGD([x], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss = 0.5 * ((x - target) ** 2).sum()
            loss.backward()
            opt.step()
        assert torch.allclose(x.detach(), target, atol=0.001)
    pass
