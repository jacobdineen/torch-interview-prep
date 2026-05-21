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
from p75b_at import *

def test_p75b_at():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    base_w_init = base.weight.detach().clone()
    base_b_init = base.bias.detach().clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    pass
    x = torch.randn(3, 16)
    with step('at init, LoRA forward equals base forward'):
        assert torch.allclose(lora(x), base(x), atol=1e-06)
    pass
    pass
    with torch.no_grad():
        lora.lora_B.fill_(0.1)
    out = lora(x)
    pass
    pass
    pass
    pass
    pass
