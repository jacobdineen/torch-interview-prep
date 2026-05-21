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
from p25e_normalize import *

def test_p25e_normalize():
    torch.manual_seed(0)
    D = 8
    rn = RMSNorm(D)
    pass
    pass
    pass
    pass
    with step('normalize over last-two dims (tuple normalized_shape)'):
        rn2 = RMSNorm((4, D))
        x2 = torch.randn(3, 4, D)
        out2 = rn2(x2)
        assert out2.shape == x2.shape
        rms2 = out2.pow(2).mean(dim=(-1, -2)).sqrt()
        assert torch.allclose(rms2, torch.ones_like(rms2), atol=0.0001)
