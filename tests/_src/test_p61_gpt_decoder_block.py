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
from p61_gpt_decoder_block import *

def test_p61_gpt_decoder_block():

    torch.manual_seed(0)
    B, T, D = 2, 6, 16
    blk = GPTDecoderBlock(d_model=D, num_heads=4, d_ff=32, dropout=0.0)
    blk.eval()
    x = torch.randn(B, T, D)
    out = blk(x)
    with step("output shape matches input"):
        assert out.shape == (B, T, D)
    with step("causality: perturbing future tokens leaves earlier outputs unchanged"):
        x_perturbed = x.clone()
        x_perturbed[:, 3:] = torch.randn_like(x_perturbed[:, 3:])
        out_p = blk(x_perturbed)
        assert torch.allclose(out[:, :3], out_p[:, :3], atol=1e-5)
    with step("gradients flow into every parameter"):
        loss = out.sum(); loss.backward()
        for name, p in blk.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

