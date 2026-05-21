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
from p34a_large import *

def test_p34a_large():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    x = torch.randn(3, 8)
    y = torch.randn(3, 2)
    loss = ((model(x) - y) ** 2).sum()
    loss.backward()
    flat = torch.cat([p.grad.flatten() for p in model.parameters()])
    expected_norm = flat.norm()
    g_copies = [p.grad.clone() for p in model.parameters()]
    with step('large max_norm: gradients unchanged; returns pre-clip norm'):
        n = clip_grad_norm_(model.parameters(), max_norm=1000000.0)
        assert torch.allclose(n, expected_norm, atol=1e-06)
        for p, gc in zip(model.parameters(), g_copies):
            assert torch.equal(p.grad, gc)
    pass
