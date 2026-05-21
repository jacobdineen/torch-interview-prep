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
from p14_linear_layer_from_scratch import *

def test_p14_linear_layer_from_scratch():

    torch.manual_seed(0)
    layer = MyLinear(8, 4)
    with step("weight shape (out, in) and bias shape (out,)"):
        assert hasattr(layer, "weight") and layer.weight.shape == (4, 8)
        assert hasattr(layer, "bias") and layer.bias.shape == (4,)
    ref = nn.Linear(8, 4)
    with torch.no_grad():
        layer.weight.copy_(ref.weight); layer.bias.copy_(ref.bias)
    with step("forward on (5, 8) matches nn.Linear"):
        x = torch.randn(5, 8)
        out = layer(x)
        assert out.shape == (5, 4)
        assert torch.allclose(out, ref(x), atol=1e-6)
    with step("forward on (2, 6, 8) batched input matches nn.Linear"):
        x3 = torch.randn(2, 6, 8)
        out3 = layer(x3)
        assert out3.shape == (2, 6, 4)
        assert torch.allclose(out3, ref(x3), atol=1e-6)
    with step("bias=False mode: forward still produces (5, 4)"):
        layer_nb = MyLinear(8, 4, bias=False)
        out_nb = layer_nb(torch.randn(5, 8))
        assert out_nb.shape == (5, 4)

