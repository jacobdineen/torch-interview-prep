import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import torch
from p31_adam_optimizer import *

def test_p31_adam_optimizer():

    torch.manual_seed(0)
    with step("Adam matches torch.optim.Adam over 30 steps"):
        a = torch.randn(8, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_(True)
        target = torch.randn(8)
        mine = MyAdam([a], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
        ref = torch.optim.Adam([a_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-3)
        for _ in range(30):
            mine.zero_grad(); ref.zero_grad()
            ((a - target) ** 2).sum().backward()
            ((a_ref - target) ** 2).sum().backward()
            mine.step(); ref.step()
        assert torch.allclose(a.detach(), a_ref.detach(), atol=1e-5)
    with step("Adam converges on a quadratic"):
        torch.manual_seed(1)
        x = torch.zeros(3, requires_grad=True)
        target = torch.tensor([1.0, -2.0, 3.0])
        opt = MyAdam([x], lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            (((x - target) ** 2).sum()).backward()
            opt.step()
        assert torch.allclose(x.detach(), target, atol=5e-3)

