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
from p14b_forward import *

def test_p14b_forward():
    torch.manual_seed(0)
    layer = MyLinear(8, 4)
    pass
    ref = nn.Linear(8, 4)
    with torch.no_grad():
        layer.weight.copy_(ref.weight)
        layer.bias.copy_(ref.bias)
    with step('forward on (5, 8) matches nn.Linear'):
        x = torch.randn(5, 8)
        out = layer(x)
        assert out.shape == (5, 4)
        assert torch.allclose(out, ref(x), atol=1e-06)
    pass
    pass
