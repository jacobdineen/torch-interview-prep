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
from p73d_kl import *

def test_p73d_kl():
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
    with step('KL at (mu=0, logvar=0) is exactly 0'):
        zmu = torch.zeros(4, 3)
        zlv = torch.zeros(4, 3)
        kl = kl_divergence_standard_normal(zmu, zlv, reduction='sum')
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-06)
    pass
