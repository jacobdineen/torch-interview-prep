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
from p24b_normalize2 import *

def test_p24b_normalize2():
    torch.manual_seed(0)
    pass
    with step('normalize over last two dims matches nn.LayerNorm'):
        mine2 = MyLayerNorm((4, 8))
        ref2 = nn.LayerNorm((4, 8))
        with torch.no_grad():
            mine2.weight.copy_(ref2.weight)
            mine2.bias.copy_(ref2.bias)
        x2 = torch.randn(3, 4, 8)
        assert torch.allclose(mine2(x2), ref2(x2), atol=1e-05)
    pass
