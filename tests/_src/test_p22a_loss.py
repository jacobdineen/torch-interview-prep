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
from p22a_loss import *

def test_p22a_loss():
    torch.manual_seed(0)
    with step('loss is 0 when negative is far from anchor'):
        a = torch.zeros(4, 3)
        p = torch.zeros(4, 3)
        n = torch.full((4, 3), 5.0)
        assert triplet_margin_loss(a, p, n, margin=1.0).item() == 0.0
    pass
    pass
    pass
    pass
