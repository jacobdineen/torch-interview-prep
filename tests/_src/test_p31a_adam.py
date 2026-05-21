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
from p31a_adam import *

def test_p31a_adam():
    torch.manual_seed(0)
    with step('Adam matches torch.optim.Adam over 30 steps'):
        a = torch.randn(8, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_(True)
        target = torch.randn(8)
        mine = MyAdam([a], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
        ref = torch.optim.Adam([a_ref], lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001)
        for _ in range(30):
            mine.zero_grad()
            ref.zero_grad()
            ((a - target) ** 2).sum().backward()
            ((a_ref - target) ** 2).sum().backward()
            mine.step()
            ref.step()
        assert torch.allclose(a.detach(), a_ref.detach(), atol=1e-05)
    pass
