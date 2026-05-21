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
from p27b_train2 import *

def test_p27b_train2():
    torch.manual_seed(0)
    p = 0.4
    d = MyDropout(p)
    x = torch.ones(10000)
    pass
    with step('train: survivors scaled by 1/(1-p)'):
        d.train()
        y = d(x)
        surv = y[y != 0]
        assert torch.allclose(surv, torch.full_like(surv, 1.0 / (1 - p)), atol=1e-06)
        assert abs(y.mean().item() - 1.0) < 0.03
    pass
    pass
