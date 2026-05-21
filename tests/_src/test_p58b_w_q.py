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
from p58b_w_q import *

def test_p58b_w_q():
    torch.manual_seed(0)
    D, H, G = (32, 8, 4)
    gqa = GroupedQueryAttention(D, num_q_heads=H, num_kv_groups=G, bias=False)
    x = torch.randn(2, 5, D)
    out = gqa(x)
    head_dim = D // H
    pass
    with step('w_q weight is (H*head_dim, D); w_k/w_v are (G*head_dim, D)'):
        assert gqa.w_q.weight.shape == (H * head_dim, D)
        assert gqa.w_k.weight.shape == (G * head_dim, D)
        assert gqa.w_v.weight.shape == (G * head_dim, D)
        assert gqa.w_o.weight.shape == (D, H * head_dim)
    pass
    pass
    pass
    pass
