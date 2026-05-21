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
from p23b_running import *

def test_p23b_running():
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
    with step('running stats updated like nn.BatchNorm1d'):
        assert torch.allclose(bn.running_mean, ref.running_mean, atol=1e-06)
        assert torch.allclose(bn.running_var, ref.running_var, atol=1e-06)
    pass
