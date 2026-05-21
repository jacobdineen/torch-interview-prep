import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import torch
from p59_transformer_encoder_block import *

def test_p59_transformer_encoder_block():

    torch.manual_seed(0)
    B, T, D = 2, 6, 16
    block = TransformerEncoderBlock(d_model=D, num_heads=4, d_ff=32, dropout=0.0)
    block.eval()
    x = torch.randn(B, T, D)
    out = block(x)
    with step("output shape matches input"):
        assert out.shape == (B, T, D)
    with step("output is non-trivially different from input"):
        assert not torch.allclose(out, x, atol=1e-3)
        assert out.abs().max() > 1e-3
    with step("gradients flow into every parameter"):
        loss = out.sum(); loss.backward()
        for name, p in block.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    with step("accepts an attn_mask"):
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        out_m = block(x, attn_mask=mask)
        assert out_m.shape == (B, T, D)

