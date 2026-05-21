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
from p23c_eval import *

def test_p23c_eval():
    torch.manual_seed(0)
    C = 8
    bn = MyBatchNorm1d(C)
    ref = nn.BatchNorm1d(C)
    with torch.no_grad():
        if bn.weight is not None:
            bn.weight.copy_(ref.weight)
            bn.bias.copy_(ref.bias)
        bn.running_mean.copy_(ref.running_mean)
        bn.running_var.copy_(ref.running_var)
    pass
    pass
    with step('eval-mode uses running stats (matches nn.BatchNorm1d)'):
        bn.eval()
        ref.eval()
        x2 = torch.randn(4, C)
        assert torch.allclose(bn(x2), ref(x2), atol=1e-05)
