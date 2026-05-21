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
import torch.nn.functional as F
from p60b_output2 import *

def test_p60b_output2():
    torch.manual_seed(0)
    D, FF = (16, 32)
    blk = SwiGLU(D, FF)
    x = torch.randn(2, 7, D)
    out = blk(x)
    pass
    with step('output equals w_down(silu(w_gate(x)) * w_up(x))'):
        with torch.no_grad():
            g = F.linear(x, blk.w_gate.weight)
            u = F.linear(x, blk.w_up.weight)
            manual = F.linear(F.silu(g) * u, blk.w_down.weight)
        assert torch.allclose(out, manual, atol=1e-05)
    pass
    pass
