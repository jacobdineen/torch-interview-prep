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
from p33d_at import *

def test_p33d_at():
    pass
    pass
    pass
    with step('at total_steps -> min_lr'):
        assert math.isclose(linear_warmup_cosine_lr(100, 1.0, 10, 100, min_lr=0.1), 0.1, abs_tol=1e-09)
    pass
    pass
