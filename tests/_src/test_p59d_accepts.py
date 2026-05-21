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
from p59d_accepts import *

def test_p59d_accepts():
    torch.manual_seed(0)
    B, T, D = (2, 6, 16)
    block = TransformerEncoderBlock(d_model=D, num_heads=4, d_ff=32, dropout=0.0)
    block.eval()
    x = torch.randn(B, T, D)
    out = block(x)
    pass
    pass
    pass
    with step('accepts an attn_mask'):
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out_m = block(x, attn_mask=mask)
        assert out_m.shape == (B, T, D)
