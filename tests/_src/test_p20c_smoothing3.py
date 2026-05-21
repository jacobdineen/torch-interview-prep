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
from p20c_smoothing3 import *

def test_p20c_smoothing3():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    pass
    pass
    with step('smoothing=1.0 equals -mean(log_softmax)'):
        log_probs = F.log_softmax(logits, dim=-1)
        expected = -log_probs.mean(dim=-1).mean()
        got = label_smoothing_ce(logits, targets, smoothing=1.0)
        assert torch.allclose(got, expected, atol=1e-06)
    pass
