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
from p26c_affine import *

def test_p26c_affine():
    torch.manual_seed(0)
    G, C = (4, 8)
    pass
    pass
    with step('affine=False matches nn.GroupNorm(affine=False)'):
        mine_na = MyGroupNorm(G, C, affine=False)
        ref_na = nn.GroupNorm(G, C, affine=False)
        z = torch.randn(2, C, 4, 4)
        assert torch.allclose(mine_na(z), ref_na(z), atol=1e-05)
