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
from p12b_gradcheck import *

def test_p12b_gradcheck():
    torch.manual_seed(0)
    pass
    with step('gradcheck passes (double precision)'):
        x2 = torch.randn(5, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(stable_softplus, (x2,), eps=1e-06, atol=0.0001)
    pass
