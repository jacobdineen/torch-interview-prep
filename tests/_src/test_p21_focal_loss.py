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
from p21_focal_loss import *

def test_p21_focal_loss():

    torch.manual_seed(0)
    logits = torch.randn(6, 4) * 2
    targets = torch.tensor([0, 1, 2, 3, 1, 2])
    with step("gamma=0 equals cross-entropy"):
        expected_ce = F.cross_entropy(logits, targets, reduction="mean")
        got = focal_loss(logits, targets, gamma=0.0)
        assert torch.allclose(got, expected_ce, atol=1e-6)
    with step("gamma > 0 shrinks loss relative to CE"):
        expected_ce = F.cross_entropy(logits, targets, reduction="mean")
        fl = focal_loss(logits, targets, gamma=2.0, reduction="mean")
        assert fl.item() <= expected_ce.item() + 1e-6
    with step("alpha=[0.25]*4 scales loss by 0.25 vs alpha=None"):
        alpha = torch.tensor([0.25, 0.25, 0.25, 0.25])
        fl_a = focal_loss(logits, targets, gamma=2.0, alpha=alpha, reduction="mean")
        fl_nb = focal_loss(logits, targets, gamma=2.0, alpha=None, reduction="mean")
        assert torch.allclose(fl_a, 0.25 * fl_nb, atol=1e-6)
    with step("reduction modes consistent"):
        fl_none = focal_loss(logits, targets, gamma=2.0, reduction="none")
        assert fl_none.shape == (6,)
        fl_sum = focal_loss(logits, targets, gamma=2.0, reduction="sum")
        assert torch.allclose(fl_sum, fl_none.sum(), atol=1e-6)

