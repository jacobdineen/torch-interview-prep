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
from p52b_pe import *

def test_p52b_pe():
    PE = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    pass
    with step('PE[0] is sin(0)=0 at even cols and cos(0)=1 at odd cols'):
        even = PE[0, 0::2]
        odd = PE[0, 1::2]
        assert torch.allclose(even, torch.zeros_like(even), atol=1e-06)
        assert torch.allclose(odd, torch.ones_like(odd), atol=1e-06)
    pass
    pass
