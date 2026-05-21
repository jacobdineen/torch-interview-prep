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
import torch.nn as nn
from p13_detach_no_grad import *

def test_p13_detach_no_grad():

    torch.manual_seed(0)
    with step("param_norm matches the L2 norm of all params, no grad"):
        model = nn.Linear(4, 2)
        n = param_norm(model)
        assert isinstance(n, torch.Tensor)
        assert not n.requires_grad
        expected = torch.cat([p.flatten() for p in model.parameters()]).norm()
        assert torch.isclose(n, expected, atol=1e-6)
    with step("stop_gradient_at: forward equals x; grads blocked above threshold"):
        x = torch.tensor([0.5, 1.0, 2.0, 3.0], requires_grad=True)
        y = stop_gradient_at(x, threshold=1.5)
        assert torch.allclose(y, x.detach())
        s = y.sum(); s.backward()
        assert torch.equal(x.grad, torch.tensor([1.0, 1.0, 0.0, 0.0]))
    with step("is_leaf_and_requires_grad reports the right flags"):
        a = torch.randn(3, requires_grad=True)
        leaf, rg = is_leaf_and_requires_grad(a)
        assert leaf is True and rg is True
        b = (a * 2)
        leaf2, rg2 = is_leaf_and_requires_grad(b)
        assert leaf2 is False and rg2 is True

