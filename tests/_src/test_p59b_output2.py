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
from p59b_output2 import *

def test_p59b_output2():
    torch.manual_seed(0)
    B, T, D = (2, 6, 16)
    block = TransformerEncoderBlock(d_model=D, num_heads=4, d_ff=32, dropout=0.0)
    block.eval()
    x = torch.randn(B, T, D)
    out = block(x)
    pass
    with step('output is non-trivially different from input'):
        assert not torch.allclose(out, x, atol=0.001)
        assert out.abs().max() > 0.001
    pass
    pass
