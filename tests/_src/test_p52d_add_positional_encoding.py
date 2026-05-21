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
from p52d_add_positional_encoding import *

def test_p52d_add_positional_encoding():
    PE = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    pass
    pass
    pass
    with step('add_positional_encoding adds PE across the batch'):
        x = torch.zeros(2, 10, 16)
        y = add_positional_encoding(x)
        assert torch.allclose(y[0], PE)
        assert torch.allclose(y[1], PE)
