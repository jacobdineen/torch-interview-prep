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
from p20d_reduction import *

def test_p20d_reduction():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    pass
    pass
    pass
    with step("reduction='none' returns per-sample loss"):
        got_none = label_smoothing_ce(logits, targets, smoothing=0.1, reduction='none')
        assert got_none.shape == (8,)
        got_sum = label_smoothing_ce(logits, targets, smoothing=0.1, reduction='sum')
        assert torch.isclose(got_sum, got_none.sum(), atol=1e-06)
