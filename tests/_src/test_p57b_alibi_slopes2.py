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
from p57b_alibi_slopes2 import *

def test_p57b_alibi_slopes2():
    H = 4
    slopes = alibi_slopes(H)
    pass
    with step('alibi_slopes match m_h = 2 ** (-8 * h / H) for h=1..H'):
        expected = torch.tensor([2.0 ** (-8.0 * (i + 1) / H) for i in range(H)])
        assert torch.allclose(slopes, expected, atol=1e-06)
    T = 6
    bias = alibi_bias(H, T, causal=False)
    pass
    pass
    pass
    pass
    bias_c = alibi_bias(H, T, causal=True)
    pass
