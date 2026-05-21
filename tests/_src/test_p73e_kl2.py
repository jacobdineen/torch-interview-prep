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
from p73e_kl2 import *

def test_p73e_kl2():
    torch.manual_seed(0)
    mu = torch.randn(8, 4, requires_grad=True)
    logvar = torch.randn(8, 4, requires_grad=True)
    samples = []
    for s in range(1000):
        g = torch.Generator().manual_seed(s)
        samples.append(reparameterize(mu.detach(), logvar.detach(), generator=g))
    samples = torch.stack(samples)
    pass
    pass
    pass
    pass
    with step('KL elementwise matches the closed form'):
        mu2 = torch.randn(5, 3)
        lv2 = torch.randn(5, 3)
        closed = 0.5 * (mu2.pow(2) + lv2.exp() - 1 - lv2)
        assert torch.allclose(kl_divergence_standard_normal(mu2, lv2, reduction='none'), closed, atol=1e-06)
        assert torch.allclose(kl_divergence_standard_normal(mu2, lv2, reduction='sum'), closed.sum(), atol=1e-05)
        assert torch.allclose(kl_divergence_standard_normal(mu2, lv2, reduction='batchmean'), closed.sum() / 5, atol=1e-05)
