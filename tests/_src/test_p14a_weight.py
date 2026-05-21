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
from p14a_weight import *

def test_p14a_weight():
    torch.manual_seed(0)
    layer = MyLinear(8, 4)
    with step('weight shape (out, in) and bias shape (out,)'):
        assert hasattr(layer, 'weight') and layer.weight.shape == (4, 8)
        assert hasattr(layer, 'bias') and layer.bias.shape == (4,)
    ref = nn.Linear(8, 4)
    with torch.no_grad():
        layer.weight.copy_(ref.weight)
        layer.bias.copy_(ref.bias)
    pass
    pass
    pass
