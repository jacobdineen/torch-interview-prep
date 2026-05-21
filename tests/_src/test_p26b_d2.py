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
from p26b_d2 import *

def test_p26b_d2():
    torch.manual_seed(0)
    G, C = (4, 8)
    pass
    with step('3D input (N,C,T) matches nn.GroupNorm'):
        mine2 = MyGroupNorm(G, C)
        ref2 = nn.GroupNorm(G, C)
        if mine2.weight is not None:
            with torch.no_grad():
                mine2.weight.copy_(ref2.weight)
                mine2.bias.copy_(ref2.bias)
        y = torch.randn(3, C, 10)
        assert torch.allclose(mine2(y), ref2(y), atol=1e-05)
    pass
