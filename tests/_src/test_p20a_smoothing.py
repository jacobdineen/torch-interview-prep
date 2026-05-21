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
from p20a_smoothing import *

def test_p20a_smoothing():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    with step('smoothing=0 equals plain cross-entropy'):
        expected = F.cross_entropy(logits, targets)
        got = label_smoothing_ce(logits, targets, smoothing=0.0)
        assert torch.allclose(got, expected, atol=1e-06)
    pass
    pass
    pass
