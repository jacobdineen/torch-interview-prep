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
import torch.nn.functional as F
from p71_info_nce_loss import *

def test_p71_info_nce_loss():

    torch.manual_seed(0)
    N, D = 64, 32
    a = torch.randn(N, D)
    with step("identical views: loss is near zero"):
        loss = info_nce_loss(a, a, tau=0.1)
        assert loss.item() < 0.05
    with step("uncorrelated views: loss is near log(N)"):
        b = torch.randn(N, D)
        loss = info_nce_loss(a, b, tau=1.0)
        target = math.log(N)
        assert abs(loss.item() - target) < 1.0
    with step("gradients flow through both views"):
        a_g = torch.randn(N, D, requires_grad=True)
        b_g = torch.randn(N, D, requires_grad=True)
        l = info_nce_loss(a_g, b_g, tau=0.1)
        l.backward()
        assert a_g.grad is not None and torch.isfinite(a_g.grad).all()
        assert b_g.grad is not None and torch.isfinite(b_g.grad).all()
    with step("matches the symmetric cross-entropy reference"):
        b = torch.randn(N, D)
        a_n = F.normalize(a, dim=-1); b_n = F.normalize(b, dim=-1)
        logits = a_n @ b_n.T
        labels = torch.arange(N)
        expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        got = info_nce_loss(a, b, tau=1.0)
        assert torch.allclose(got, expected, atol=1e-5)

