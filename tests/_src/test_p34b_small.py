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
import torch.nn as nn
from p34b_small import *

def test_p34b_small():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    x = torch.randn(3, 8)
    y = torch.randn(3, 2)
    loss = ((model(x) - y) ** 2).sum()
    loss.backward()
    flat = torch.cat([p.grad.flatten() for p in model.parameters()])
    expected_norm = flat.norm()
    g_copies = [p.grad.clone() for p in model.parameters()]
    pass
    with step('small max_norm: grad norm clipped, return value is pre-clip norm'):
        target_norm = 0.1
        n2 = clip_grad_norm_(model.parameters(), max_norm=target_norm)
        assert torch.allclose(n2, expected_norm, atol=1e-06)
        flat2 = torch.cat([p.grad.flatten() for p in model.parameters()])
        assert flat2.norm().item() <= target_norm + 0.0001
