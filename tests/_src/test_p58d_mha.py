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
from p58d_mha import *

def test_p58d_mha():
    torch.manual_seed(0)
    D, H, G = (32, 8, 4)
    gqa = GroupedQueryAttention(D, num_q_heads=H, num_kv_groups=G, bias=False)
    x = torch.randn(2, 5, D)
    out = gqa(x)
    head_dim = D // H
    pass
    pass
    pass
    with step('MHA degenerate (G=H) still works'):
        mha_like = GroupedQueryAttention(D, num_q_heads=H, num_kv_groups=H, bias=False)
        assert mha_like(x).shape == (2, 5, D)
    pass
    pass
