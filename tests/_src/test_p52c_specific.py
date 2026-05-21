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
from p52c_specific import *

def test_p52c_specific():
    PE = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    pass
    pass
    with step('specific entry matches the formula'):
        pos, i, d_model = (3, 2, 16)
        expected_even = math.sin(pos / 10000 ** (2 * i / d_model))
        expected_odd = math.cos(pos / 10000 ** (2 * i / d_model))
        assert math.isclose(PE[pos, 2 * i].item(), expected_even, abs_tol=1e-06)
        assert math.isclose(PE[pos, 2 * i + 1].item(), expected_odd, abs_tol=1e-06)
    pass
