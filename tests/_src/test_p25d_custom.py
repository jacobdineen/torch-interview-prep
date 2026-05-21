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
from p25d_custom import *

def test_p25d_custom():
    torch.manual_seed(0)
    D = 8
    rn = RMSNorm(D)
    pass
    pass
    pass
    with step('custom weight scales the RMS'):
        with torch.no_grad():
            rn.weight.fill_(2.0)
        x = torch.randn(2, 5, D)
        out = rn(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, 2 * torch.ones_like(rms), atol=0.0001)
    pass
