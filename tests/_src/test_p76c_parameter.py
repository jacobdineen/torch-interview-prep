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
import copy
import torch
import torch.nn as nn
from p76c_parameter import *

def test_p76c_parameter():
    torch.manual_seed(0)
    blocks = [nn.Linear(8, 8) for _ in range(4)]
    ref = nn.Sequential(*blocks)
    ck = CheckpointedSequential(copy.deepcopy(list(blocks)))
    for p_ref, p_ck in zip(ref.parameters(), ck.parameters()):
        with torch.no_grad():
            p_ck.copy_(p_ref)
    x = torch.randn(2, 8, requires_grad=True)
    x_ck = x.detach().clone().requires_grad_(True)
    out_ref = ref(x)
    out_ck = ck(x_ck)
    pass
    out_ref.sum().backward()
    out_ck.sum().backward()
    pass
    with step('parameter gradients per-block match reference'):
        for p_ref, p_ck in zip(ref.parameters(), ck.parameters()):
            assert torch.allclose(p_ref.grad, p_ck.grad, atol=1e-06)
    pass
