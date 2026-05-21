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
from p17_mse_huber_loss import *

def test_p17_mse_huber_loss():

    torch.manual_seed(0)
    p = torch.randn(8, 5); t = torch.randn(8, 5)
    with step("mse_loss (mean) matches F.mse_loss"):
        assert torch.allclose(mse_loss(p, t), torch.nn.functional.mse_loss(p, t), atol=1e-6)
    with step("mse_loss (sum) matches"):
        assert torch.allclose(mse_loss(p, t, reduction="sum"),
                              torch.nn.functional.mse_loss(p, t, reduction="sum"), atol=1e-5)
    with step("mse_loss (none) returns elementwise"):
        assert torch.allclose(mse_loss(p, t, reduction="none"),
                              torch.nn.functional.mse_loss(p, t, reduction="none"), atol=1e-6)
    with step("huber_loss with delta=0.7 matches F.huber_loss"):
        expected = torch.nn.functional.huber_loss(p, t, delta=0.7, reduction="mean")
        assert torch.allclose(huber_loss(p, t, delta=0.7), expected, atol=1e-6)
    with step("huber_loss (none) elementwise matches"):
        expected = torch.nn.functional.huber_loss(p, t, delta=0.7, reduction="none")
        assert torch.allclose(huber_loss(p, t, delta=0.7, reduction="none"), expected, atol=1e-6)

