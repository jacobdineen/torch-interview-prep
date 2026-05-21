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
from p21d_reduction import *

def test_p21d_reduction():
    torch.manual_seed(0)
    logits = torch.randn(6, 4) * 2
    targets = torch.tensor([0, 1, 2, 3, 1, 2])
    pass
    pass
    pass
    with step('reduction modes consistent'):
        fl_none = focal_loss(logits, targets, gamma=2.0, reduction='none')
        assert fl_none.shape == (6,)
        fl_sum = focal_loss(logits, targets, gamma=2.0, reduction='sum')
        assert torch.allclose(fl_sum, fl_none.sum(), atol=1e-06)
