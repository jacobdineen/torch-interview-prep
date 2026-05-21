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
from p71b_uncorrelated import *

def test_p71b_uncorrelated():
    torch.manual_seed(0)
    N, D = (64, 32)
    a = torch.randn(N, D)
    pass
    with step('uncorrelated views: loss is near log(N)'):
        b = torch.randn(N, D)
        loss = info_nce_loss(a, b, tau=1.0)
        target = math.log(N)
        assert abs(loss.item() - target) < 1.0
    pass
    pass
