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
from p24a_normalize import *

def test_p24a_normalize():
    torch.manual_seed(0)
    with step('normalize over last dim only matches nn.LayerNorm'):
        mine = MyLayerNorm(8)
        ref = nn.LayerNorm(8)
        with torch.no_grad():
            mine.weight.copy_(ref.weight)
            mine.bias.copy_(ref.bias)
        x = torch.randn(2, 5, 8)
        assert torch.allclose(mine(x), ref(x), atol=1e-05)
    pass
    pass
