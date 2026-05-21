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
from p73b_empirical2 import *

def test_p73b_empirical2():
    torch.manual_seed(0)
    mu = torch.randn(8, 4, requires_grad=True)
    logvar = torch.randn(8, 4, requires_grad=True)
    samples = []
    for s in range(1000):
        g = torch.Generator().manual_seed(s)
        samples.append(reparameterize(mu.detach(), logvar.detach(), generator=g))
    samples = torch.stack(samples)
    pass
    with step('empirical std ~ exp(0.5 * logvar)'):
        emp_std = samples.std(dim=0, unbiased=False)
        expected_std = torch.exp(0.5 * logvar.detach())
        rel = (emp_std - expected_std).abs() / (expected_std + 0.001)
        assert rel.max().item() < 0.25
    pass
    pass
    pass
