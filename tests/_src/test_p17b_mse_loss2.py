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
from p17b_mse_loss2 import *

def test_p17b_mse_loss2():
    torch.manual_seed(0)
    p = torch.randn(8, 5)
    t = torch.randn(8, 5)
    pass
    with step('mse_loss (sum) matches'):
        assert torch.allclose(mse_loss(p, t, reduction='sum'), torch.nn.functional.mse_loss(p, t, reduction='sum'), atol=1e-05)
    pass
    pass
    pass
