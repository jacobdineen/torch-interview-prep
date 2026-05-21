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
from p12_custom_autograd_function import *

def test_p12_custom_autograd_function():

    torch.manual_seed(0)
    with step("forward matches torch.nn.functional.softplus over a wide range"):
        x = torch.linspace(-50, 50, steps=11)
        y = stable_softplus(x)
        expected = torch.nn.functional.softplus(x)
        assert torch.allclose(y, expected, atol=1e-5)
    with step("gradcheck passes (double precision)"):
        x2 = torch.randn(5, dtype=torch.double, requires_grad=True)
        assert torch.autograd.gradcheck(stable_softplus, (x2,), eps=1e-6, atol=1e-4)
    with step("gradient equals sigmoid(x)"):
        x3 = torch.randn(4, requires_grad=True)
        out = stable_softplus(x3).sum()
        out.backward()
        assert torch.allclose(x3.grad, torch.sigmoid(x3), atol=1e-5)

