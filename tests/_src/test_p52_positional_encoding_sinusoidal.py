import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import math
import torch
from p52_positional_encoding_sinusoidal import *

def test_p52_positional_encoding_sinusoidal():

    PE = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    with step("shape is (seq_len, d_model)"):
        assert PE.shape == (10, 16)
    with step("PE[0] is sin(0)=0 at even cols and cos(0)=1 at odd cols"):
        even = PE[0, 0::2]; odd = PE[0, 1::2]
        assert torch.allclose(even, torch.zeros_like(even), atol=1e-6)
        assert torch.allclose(odd, torch.ones_like(odd), atol=1e-6)
    with step("specific entry matches the formula"):
        pos, i, d_model = 3, 2, 16
        expected_even = math.sin(pos / (10000 ** (2 * i / d_model)))
        expected_odd = math.cos(pos / (10000 ** (2 * i / d_model)))
        assert math.isclose(PE[pos, 2 * i].item(), expected_even, abs_tol=1e-6)
        assert math.isclose(PE[pos, 2 * i + 1].item(), expected_odd, abs_tol=1e-6)
    with step("add_positional_encoding adds PE across the batch"):
        x = torch.zeros(2, 10, 16)
        y = add_positional_encoding(x)
        assert torch.allclose(y[0], PE)
        assert torch.allclose(y[1], PE)

