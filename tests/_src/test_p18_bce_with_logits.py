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
import torch.nn.functional as F
from p18_bce_with_logits import *

def test_p18_bce_with_logits():

    torch.manual_seed(0)
    logits = torch.randn(10, 3) * 4
    targets = (torch.rand(10, 3) > 0.5).float()
    for red in ("mean", "sum", "none"):
        with step(f"bce_with_logits matches F.binary_cross_entropy_with_logits (reduction={red!r})"):
            expected = F.binary_cross_entropy_with_logits(logits, targets, reduction=red)
            got = bce_with_logits(logits, targets, reduction=red)
            assert torch.allclose(got, expected, atol=1e-6)
    with step("pos_weight scales positive terms"):
        pw = torch.tensor([2.0, 0.5, 1.0])
        expected = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
        got = bce_with_logits(logits, targets, pos_weight=pw)
        assert torch.allclose(got, expected, atol=1e-6)
    with step("bce_with_logits is finite at extreme logits"):
        big = torch.tensor([[-1e4, 0.0, 1e4]])
        tgt = torch.tensor([[0.0, 1.0, 1.0]])
        out = bce_with_logits(big, tgt)
        assert torch.isfinite(out).all()

