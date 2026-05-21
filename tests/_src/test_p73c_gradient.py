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
from p73c_gradient import *

def test_p73c_gradient():
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
    with step('gradient flows through the reparameterization'):
        g = torch.Generator().manual_seed(42)
        z = reparameterize(mu, logvar, generator=g)
        z.sum().backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert logvar.grad is not None and torch.isfinite(logvar.grad).all()
    pass
    pass
