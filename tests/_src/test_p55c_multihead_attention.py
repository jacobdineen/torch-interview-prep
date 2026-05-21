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
from p55c_multihead_attention import *

def test_p55c_multihead_attention():
    torch.manual_seed(0)
    B, T, D, H = (2, 7, 16, 4)
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    out = mha(x)
    pass
    pass
    with step('matches nn.MultiheadAttention when weights are copied'):
        ref = nn.MultiheadAttention(D, H, batch_first=True, bias=True)
        with torch.no_grad():
            Wq, Wk, Wv = ref.in_proj_weight.chunk(3, dim=0)
            bq, bk, bv = ref.in_proj_bias.chunk(3, dim=0)
            mha.w_q.weight.copy_(Wq)
            mha.w_q.bias.copy_(bq)
            mha.w_k.weight.copy_(Wk)
            mha.w_k.bias.copy_(bk)
            mha.w_v.weight.copy_(Wv)
            mha.w_v.bias.copy_(bv)
            mha.w_o.weight.copy_(ref.out_proj.weight)
            mha.w_o.bias.copy_(ref.out_proj.bias)
        ref_out, _ = ref(x, x, x, need_weights=False)
        mine_out = mha(x)
        assert torch.allclose(mine_out, ref_out, atol=0.0001)
    pass
