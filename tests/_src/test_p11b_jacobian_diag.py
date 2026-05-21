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
from p11b_jacobian_diag import *

def test_p11b_jacobian_diag():
    torch.manual_seed(0)
    pass
    with step('jacobian_diag of f(z)=z^3 equals 3*z^2'):

        def f(z):
            return z ** 3
        z = torch.tensor([2.0, -1.0, 4.0])
        jd = jacobian_diag(f, z)
        assert torch.allclose(jd, 3 * z ** 2)
    pass
