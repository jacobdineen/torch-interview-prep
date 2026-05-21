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
from p75d_trainable import *

def test_p75d_trainable():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    base_w_init = base.weight.detach().clone()
    base_b_init = base.bias.detach().clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    pass
    x = torch.randn(3, 16)
    pass
    pass
    with step('trainable parameter count is r * (in + out)'):
        trainable = sum((p.numel() for p in lora.parameters() if p.requires_grad))
        assert trainable == 4 * (16 + 8)
    with torch.no_grad():
        lora.lora_B.fill_(0.1)
    out = lora(x)
    pass
    pass
    pass
    pass
    pass
