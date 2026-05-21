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
from p24_layernorm import *

def test_p24_layernorm():

    torch.manual_seed(0)
    with step("normalize over last dim only matches nn.LayerNorm"):
        mine = MyLayerNorm(8); ref = nn.LayerNorm(8)
        with torch.no_grad():
            mine.weight.copy_(ref.weight); mine.bias.copy_(ref.bias)
        x = torch.randn(2, 5, 8)
        assert torch.allclose(mine(x), ref(x), atol=1e-5)
    with step("normalize over last two dims matches nn.LayerNorm"):
        mine2 = MyLayerNorm((4, 8)); ref2 = nn.LayerNorm((4, 8))
        with torch.no_grad():
            mine2.weight.copy_(ref2.weight); mine2.bias.copy_(ref2.bias)
        x2 = torch.randn(3, 4, 8)
        assert torch.allclose(mine2(x2), ref2(x2), atol=1e-5)
    with step("elementwise_affine=False also matches"):
        mine3 = MyLayerNorm(6, elementwise_affine=False)
        ref3 = nn.LayerNorm(6, elementwise_affine=False)
        x3 = torch.randn(2, 6)
        assert torch.allclose(mine3(x3), ref3(x3), atol=1e-5)

