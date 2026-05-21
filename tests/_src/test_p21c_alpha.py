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
from p21c_alpha import *

def test_p21c_alpha():
    torch.manual_seed(0)
    logits = torch.randn(6, 4) * 2
    targets = torch.tensor([0, 1, 2, 3, 1, 2])
    pass
    pass
    with step('alpha=[0.25]*4 scales loss by 0.25 vs alpha=None'):
        alpha = torch.tensor([0.25, 0.25, 0.25, 0.25])
        fl_a = focal_loss(logits, targets, gamma=2.0, alpha=alpha, reduction='mean')
        fl_nb = focal_loss(logits, targets, gamma=2.0, alpha=None, reduction='mean')
        assert torch.allclose(fl_a, 0.25 * fl_nb, atol=1e-06)
    pass
