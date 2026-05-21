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
from p14d_bias import *

def test_p14d_bias():
    torch.manual_seed(0)
    layer = MyLinear(8, 4)
    pass
    ref = nn.Linear(8, 4)
    with torch.no_grad():
        layer.weight.copy_(ref.weight)
        layer.bias.copy_(ref.bias)
    pass
    pass
    with step('bias=False mode: forward still produces (5, 4)'):
        layer_nb = MyLinear(8, 4, bias=False)
        out_nb = layer_nb(torch.randn(5, 8))
        assert out_nb.shape == (5, 4)
