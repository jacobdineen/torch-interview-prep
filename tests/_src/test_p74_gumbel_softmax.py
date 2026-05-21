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

import math
import torch
from p74_gumbel_softmax import *

def test_p74_gumbel_softmax():

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    samples = gumbel_noise((100000,), generator=g)
    with step("gumbel_noise produces finite values"):
        assert torch.isfinite(samples).all()
    with step("gumbel_noise empirical mean ~ Euler-Mascheroni (0.5772)"):
        assert abs(samples.mean().item() - 0.5772) < 0.05
    with step("gumbel_noise empirical variance ~ pi^2 / 6"):
        assert abs(samples.var().item() - (math.pi ** 2) / 6) < 0.2
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    with step("soft gumbel_softmax produces probability simplex (rows sum to 1)"):
        g2 = torch.Generator().manual_seed(0)
        y = gumbel_softmax(logits, tau=1.0, hard=False, generator=g2)
        assert y.shape == logits.shape
        assert torch.allclose(y.sum(dim=-1), torch.ones(1), atol=1e-5)
        assert (y >= 0).all()
    with step("tau -> 0 makes the sample near one-hot"):
        g3 = torch.Generator().manual_seed(0)
        y_hot = gumbel_softmax(logits, tau=0.01, hard=False, generator=g3)
        assert y_hot.max(dim=-1).values.item() > 0.99
    with step("hard=True returns one-hot but keeps straight-through grad"):
        logits_r = torch.randn(5, 8, requires_grad=True)
        g4 = torch.Generator().manual_seed(0)
        y_hard = gumbel_softmax(logits_r, tau=1.0, hard=True, generator=g4)
        assert y_hard.shape == logits_r.shape
        assert torch.allclose(y_hard.sum(dim=-1), torch.ones(5))
        assert (y_hard == 0).logical_or(y_hard == 1).all()
        y_hard.sum().backward()
        assert logits_r.grad is not None and torch.isfinite(logits_r.grad).all()

