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
from p17a_mse_loss import *

def test_p17a_mse_loss():
    torch.manual_seed(0)
    p = torch.randn(8, 5)
    t = torch.randn(8, 5)
    with step('mse_loss (mean) matches F.mse_loss'):
        assert torch.allclose(mse_loss(p, t), torch.nn.functional.mse_loss(p, t), atol=1e-06)
    pass
    pass
    pass
    pass
