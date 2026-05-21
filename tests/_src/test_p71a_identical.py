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
import math
import torch
import torch.nn.functional as F
from p71a_identical import *

def test_p71a_identical():
    torch.manual_seed(0)
    N, D = (64, 32)
    a = torch.randn(N, D)
    with step('identical views: loss is near zero'):
        loss = info_nce_loss(a, a, tau=0.1)
        assert loss.item() < 0.05
    pass
    pass
    pass
