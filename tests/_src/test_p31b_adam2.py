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
from p31b_adam2 import *

def test_p31b_adam2():
    torch.manual_seed(0)
    pass
    with step('Adam converges on a quadratic'):
        torch.manual_seed(1)
        x = torch.zeros(3, requires_grad=True)
        target = torch.tensor([1.0, -2.0, 3.0])
        opt = MyAdam([x], lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            ((x - target) ** 2).sum().backward()
            opt.step()
        assert torch.allclose(x.detach(), target, atol=0.005)
