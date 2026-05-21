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
from p75e_non import *

def test_p75e_non():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    base_w_init = base.weight.detach().clone()
    base_b_init = base.bias.detach().clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    pass
    x = torch.randn(3, 16)
    pass
    pass
    pass
    with torch.no_grad():
        lora.lora_B.fill_(0.1)
    out = lora(x)
    with step('non-zero B makes output diverge from base output'):
        assert not torch.allclose(out, base(x), atol=0.001)
    pass
    pass
    pass
    pass
