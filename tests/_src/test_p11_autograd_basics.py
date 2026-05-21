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

import torch
from p11_autograd_basics import *

def test_p11_autograd_basics():

    torch.manual_seed(0)
    with step("grad_of_sum_of_squares matches 2*x"):
        x = torch.tensor([1.0, -2.0, 3.0])
        g = grad_of_sum_of_squares(x)
        assert torch.allclose(g, 2 * x)
    with step("jacobian_diag of f(z)=z^3 equals 3*z^2"):
        def f(z): return z ** 3
        z = torch.tensor([2.0, -1.0, 4.0])
        jd = jacobian_diag(f, z)
        assert torch.allclose(jd, 3 * z ** 2)
    with step("second_derivative of sum(x^4) equals 12*x^2"):
        x2 = torch.tensor([1.0, 2.0, 3.0])
        h = second_derivative(x2)
        assert torch.allclose(h, 12 * x2 ** 2)

