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
from p73_vae_reparameterization import *

def test_p73_vae_reparameterization():

    torch.manual_seed(0)
    mu = torch.randn(8, 4, requires_grad=True)
    logvar = torch.randn(8, 4, requires_grad=True)
    samples = []
    for s in range(1000):
        g = torch.Generator().manual_seed(s)
        samples.append(reparameterize(mu.detach(), logvar.detach(), generator=g))
    samples = torch.stack(samples)
    with step("empirical mean across many seeds ~ mu"):
        assert torch.allclose(samples.mean(dim=0), mu.detach(), atol=0.15)
    with step("empirical std ~ exp(0.5 * logvar)"):
        emp_std = samples.std(dim=0, unbiased=False)
        expected_std = torch.exp(0.5 * logvar.detach())
        rel = (emp_std - expected_std).abs() / (expected_std + 1e-3)
        assert rel.max().item() < 0.25
    with step("gradient flows through the reparameterization"):
        g = torch.Generator().manual_seed(42)
        z = reparameterize(mu, logvar, generator=g)
        z.sum().backward()
        assert mu.grad is not None and torch.isfinite(mu.grad).all()
        assert logvar.grad is not None and torch.isfinite(logvar.grad).all()
    with step("KL at (mu=0, logvar=0) is exactly 0"):
        zmu = torch.zeros(4, 3); zlv = torch.zeros(4, 3)
        kl = kl_divergence_standard_normal(zmu, zlv, reduction="sum")
        assert torch.isclose(kl, torch.tensor(0.0), atol=1e-6)
    with step("KL elementwise matches the closed form"):
        mu2 = torch.randn(5, 3); lv2 = torch.randn(5, 3)
        closed = 0.5 * (mu2.pow(2) + lv2.exp() - 1 - lv2)
        assert torch.allclose(kl_divergence_standard_normal(mu2, lv2, reduction="none"), closed, atol=1e-6)
        assert torch.allclose(kl_divergence_standard_normal(mu2, lv2, reduction="sum"), closed.sum(), atol=1e-5)
        assert torch.allclose(kl_divergence_standard_normal(mu2, lv2, reduction="batchmean"), closed.sum() / 5, atol=1e-5)

