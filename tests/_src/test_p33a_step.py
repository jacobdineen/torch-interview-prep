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
from p33a_step import *

def test_p33a_step():
    with step('step=0 -> 0 (warmup start)'):
        assert math.isclose(linear_warmup_cosine_lr(0, 1.0, 10, 100), 0.0, abs_tol=1e-09)
    pass
    pass
    pass
    pass
    pass
