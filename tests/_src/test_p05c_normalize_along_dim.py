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
from p05c_normalize_along_dim import *

def test_p05c_normalize_along_dim():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 16, 16)
    pass
    pass
    with step('normalize_along_dim subtracts mean and divides by std along dim'):
        z = torch.randn(3, 5)
        nz = normalize_along_dim(z, dim=1)
        assert torch.allclose(nz.mean(dim=1), torch.zeros(3), atol=1e-05)
        assert torch.allclose(nz.std(dim=1, unbiased=False), torch.ones(3), atol=0.0001)
    pass
