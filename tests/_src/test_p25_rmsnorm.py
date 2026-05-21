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
import torch.nn as nn
from p25_rmsnorm import *

def test_p25_rmsnorm():

    torch.manual_seed(0)
    D = 8
    rn = RMSNorm(D)
    with step("weight has shape (D,) initialized to ones"):
        assert rn.weight.shape == (D,)
        assert torch.equal(rn.weight, torch.ones(D))
    with step("output shape matches input"):
        x = torch.randn(2, 5, D)
        out = rn(x)
        assert out.shape == x.shape
    with step("with weight=1, output has unit RMS along last dim"):
        x = torch.randn(2, 5, D)
        out = rn(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    with step("custom weight scales the RMS"):
        with torch.no_grad():
            rn.weight.fill_(2.0)
        x = torch.randn(2, 5, D)
        out = rn(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, 2 * torch.ones_like(rms), atol=1e-4)
    with step("normalize over last-two dims (tuple normalized_shape)"):
        rn2 = RMSNorm((4, D))
        x2 = torch.randn(3, 4, D)
        out2 = rn2(x2)
        assert out2.shape == x2.shape
        rms2 = out2.pow(2).mean(dim=(-1, -2)).sqrt()
        assert torch.allclose(rms2, torch.ones_like(rms2), atol=1e-4)

