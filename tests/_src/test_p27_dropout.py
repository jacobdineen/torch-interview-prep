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
from p27_dropout import *

def test_p27_dropout():

    torch.manual_seed(0)
    p = 0.4
    d = MyDropout(p)
    x = torch.ones(10000)
    with step("train: drop rate is roughly p"):
        d.train()
        y = d(x)
        frac_zero = (y == 0).float().mean().item()
        assert abs(frac_zero - p) < 0.03, f"fraction zero was {frac_zero}, expected ~{p}"
    with step("train: survivors scaled by 1/(1-p)"):
        d.train()
        y = d(x)
        surv = y[y != 0]
        assert torch.allclose(surv, torch.full_like(surv, 1.0 / (1 - p)), atol=1e-6)
        assert abs(y.mean().item() - 1.0) < 0.03
    with step("eval mode is identity"):
        d.eval()
        y2 = d(x)
        assert torch.equal(y2, x)
    with step("p=0 is identity in train mode"):
        d0 = MyDropout(0.0); d0.train()
        assert torch.equal(d0(x), x)

