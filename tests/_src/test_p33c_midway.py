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
from p33c_midway import *

def test_p33c_midway():
    pass
    pass
    with step('midway cosine: progress=0.5 -> base/2'):
        val = linear_warmup_cosine_lr(55, 1.0, 10, 100, min_lr=0.0)
        assert math.isclose(val, 0.5, abs_tol=1e-09)
    pass
    pass
    pass
