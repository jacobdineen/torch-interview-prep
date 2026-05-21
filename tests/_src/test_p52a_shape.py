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
from p52a_shape import *

def test_p52a_shape():
    PE = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    with step('shape is (seq_len, d_model)'):
        assert PE.shape == (10, 16)
    pass
    pass
    pass
