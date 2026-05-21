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
from p53d_mask import *

def test_p53d_mask():
    torch.manual_seed(0)
    B, T, D = (2, 5, 4)
    q = torch.randn(B, T, D)
    k = torch.randn(B, T, D)
    v = torch.randn(B, T, D)
    out, attn = scaled_dot_product_attention(q, k, v)
    pass
    pass
    pass
    with step('mask True positions get zero attention weight'):
        mask = torch.zeros(B, T, T, dtype=torch.bool)
        mask[:, 0, T // 2:] = True
        out_m, attn_m = scaled_dot_product_attention(q, k, v, mask=mask)
        assert torch.all(attn_m[:, 0, T // 2:] == 0)
        assert torch.allclose(attn_m.sum(dim=-1), torch.ones(B, T), atol=1e-05)
    pass
