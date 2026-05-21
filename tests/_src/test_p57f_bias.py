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
from p57f_bias import *

def test_p57f_bias():
    H = 4
    slopes = alibi_slopes(H)
    pass
    pass
    T = 6
    bias = alibi_bias(H, T, causal=False)
    pass
    pass
    pass
    with step('bias magnitude at offset 3: -slope * 3'):
        assert math.isclose(bias[0, 0, 3].item(), -slopes[0].item() * 3.0, abs_tol=1e-06)
    bias_c = alibi_bias(H, T, causal=True)
    pass
