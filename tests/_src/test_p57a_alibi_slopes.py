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
from p57a_alibi_slopes import *

def test_p57a_alibi_slopes():
    H = 4
    slopes = alibi_slopes(H)
    with step('alibi_slopes shape (num_heads,)'):
        assert slopes.shape == (H,)
    pass
    T = 6
    bias = alibi_bias(H, T, causal=False)
    pass
    pass
    pass
    pass
    bias_c = alibi_bias(H, T, causal=True)
    pass
