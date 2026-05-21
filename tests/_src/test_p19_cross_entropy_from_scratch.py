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
from p19_cross_entropy_from_scratch import *

def test_p19_cross_entropy_from_scratch():

    torch.manual_seed(0)
    logits = torch.randn(7, 5) * 3
    targets = torch.randint(0, 5, (7,))
    for red in ("mean", "sum", "none"):
        with step(f"cross_entropy matches F.cross_entropy (reduction={red!r})"):
            expected = F.cross_entropy(logits, targets, reduction=red)
            got = cross_entropy(logits, targets, reduction=red)
            assert torch.allclose(got, expected, atol=1e-6)
    with step("ignore_index excludes those positions from the mean"):
        targets2 = targets.clone(); targets2[0] = -100
        expected = F.cross_entropy(logits, targets2, reduction="mean", ignore_index=-100)
        got = cross_entropy(logits, targets2, reduction="mean", ignore_index=-100)
        assert torch.allclose(got, expected, atol=1e-6)
    with step("cross_entropy stable at logits of order 1e4"):
        big = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 1.0]])
        loss = cross_entropy(big, torch.tensor([1]))
        assert torch.isfinite(loss).all()

