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
import torch.nn.functional as F
from p21a_gamma import *

def test_p21a_gamma():
    torch.manual_seed(0)
    logits = torch.randn(6, 4) * 2
    targets = torch.tensor([0, 1, 2, 3, 1, 2])
    with step('gamma=0 equals cross-entropy'):
        expected_ce = F.cross_entropy(logits, targets, reduction='mean')
        got = focal_loss(logits, targets, gamma=0.0)
        assert torch.allclose(got, expected_ce, atol=1e-06)
    pass
    pass
    pass
