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
from p15_activations import *

def test_p15_activations():

    torch.manual_seed(0)
    x = torch.randn(50) * 5
    with step("relu matches torch.relu"):
        assert torch.allclose(relu(x), torch.relu(x))
    with step("leaky_relu matches F.leaky_relu(0.1)"):
        assert torch.allclose(leaky_relu(x, 0.1), torch.nn.functional.leaky_relu(x, 0.1))
    with step("sigmoid matches torch.sigmoid"):
        assert torch.allclose(sigmoid(x), torch.sigmoid(x), atol=1e-6)
    with step("tanh matches torch.tanh"):
        assert torch.allclose(tanh(x), torch.tanh(x), atol=1e-6)
    with step("gelu matches F.gelu (erf-based exact form)"):
        assert torch.allclose(gelu(x), torch.nn.functional.gelu(x), atol=1e-5)
    with step("sigmoid is finite at extreme inputs"):
        extreme = torch.tensor([-1e4, -100.0, 0.0, 100.0, 1e4])
        s = sigmoid(extreme)
        assert torch.isfinite(s).all()
        assert s[0] == 0.0 and s[-1] == 1.0

