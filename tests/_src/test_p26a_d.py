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
from p26a_d import *

def test_p26a_d():
    torch.manual_seed(0)
    G, C = (4, 8)
    with step('4D input (N,C,H,W) matches nn.GroupNorm'):
        mine = MyGroupNorm(G, C)
        ref = nn.GroupNorm(G, C)
        if mine.weight is not None:
            with torch.no_grad():
                mine.weight.copy_(ref.weight)
                mine.bias.copy_(ref.bias)
        x = torch.randn(2, C, 5, 5)
        assert torch.allclose(mine(x), ref(x), atol=1e-05)
    pass
    pass
