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
from p24c_elementwise_affine import *

def test_p24c_elementwise_affine():
    torch.manual_seed(0)
    pass
    pass
    with step('elementwise_affine=False also matches'):
        mine3 = MyLayerNorm(6, elementwise_affine=False)
        ref3 = nn.LayerNorm(6, elementwise_affine=False)
        x3 = torch.randn(2, 6)
        assert torch.allclose(mine3(x3), ref3(x3), atol=1e-05)
