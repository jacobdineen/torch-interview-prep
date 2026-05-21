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
from p30_sgd_momentum_optimizer import *

def test_p30_sgd_momentum_optimizer():

    torch.manual_seed(0)
    target = torch.tensor([1.0, -2.0, 3.0])
    x = torch.zeros(3, requires_grad=True)
    with step("plain SGD converges to target on a quadratic"):
        opt = MySGD([x], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss = 0.5 * ((x - target) ** 2).sum()
            loss.backward(); opt.step()
        assert torch.allclose(x.detach(), target, atol=1e-3)
    with step("momentum + weight_decay matches torch.optim.SGD bit-for-bit"):
        torch.manual_seed(1)
        a = torch.randn(5, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_(True)
        target = torch.randn(5)
        mine = MySGD([a], lr=0.05, momentum=0.9, weight_decay=1e-3)
        ref = torch.optim.SGD([a_ref], lr=0.05, momentum=0.9, weight_decay=1e-3)
        for _ in range(20):
            mine.zero_grad(); ref.zero_grad()
            (0.5 * ((a - target) ** 2).sum()).backward()
            (0.5 * ((a_ref - target) ** 2).sum()).backward()
            mine.step(); ref.step()
        assert torch.allclose(a.detach(), a_ref.detach(), atol=1e-6)

