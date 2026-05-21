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
import torch
import torch.nn as nn
from p25b_output import *

def test_p25b_output():
    torch.manual_seed(0)
    D = 8
    rn = RMSNorm(D)
    pass
    with step('output shape matches input'):
        x = torch.randn(2, 5, D)
        out = rn(x)
        assert out.shape == x.shape
    pass
    pass
    pass
