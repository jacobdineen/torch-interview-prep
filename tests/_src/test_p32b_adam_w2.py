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
from p32b_adam_w2 import *

def test_p32b_adam_w2():
    torch.manual_seed(0)
    pass
    with step('AdamW (decoupled wd) differs from Adam (L2 wd) when wd > 0'):
        torch.manual_seed(1)
        p_w = torch.randn(4, requires_grad=True)
        p_a = p_w.detach().clone().requires_grad_(True)
        target = torch.randn(4)
        ow = MyAdamW([p_w], lr=0.01, weight_decay=0.1)
        oa = torch.optim.Adam([p_a], lr=0.01, weight_decay=0.1)
        for _ in range(20):
            ow.zero_grad()
            oa.zero_grad()
            ((p_w - target) ** 2).sum().backward()
            ((p_a - target) ** 2).sum().backward()
            ow.step()
            oa.step()
        assert not torch.allclose(p_w.detach(), p_a.detach(), atol=0.001)
