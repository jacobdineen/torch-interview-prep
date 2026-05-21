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
import torch.nn.functional as F
from p60_swiglu import *

def test_p60_swiglu():

    torch.manual_seed(0)
    D, FF = 16, 32
    blk = SwiGLU(D, FF)
    x = torch.randn(2, 7, D)
    out = blk(x)
    with step("output shape matches input"):
        assert out.shape == (2, 7, D)
    with step("output equals w_down(silu(w_gate(x)) * w_up(x))"):
        with torch.no_grad():
            g = F.linear(x, blk.w_gate.weight)
            u = F.linear(x, blk.w_up.weight)
            manual = F.linear(F.silu(g) * u, blk.w_down.weight)
        assert torch.allclose(out, manual, atol=1e-5)
    with step("no biases on the linears"):
        assert blk.w_gate.bias is None
        assert blk.w_up.bias is None
        assert blk.w_down.bias is None
    with step("gradients are finite for every parameter"):
        out.sum().backward()
        for name, p in blk.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name

