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
from p32_adamw_optimizer import *

def test_p32_adamw_optimizer():

    torch.manual_seed(0)
    with step("AdamW matches torch.optim.AdamW bit-for-bit"):
        a = torch.randn(8, requires_grad=True)
        a_ref = a.detach().clone().requires_grad_(True)
        target = torch.randn(8)
        mine = MyAdamW([a], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
        ref = torch.optim.AdamW([a_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
        for _ in range(50):
            mine.zero_grad(); ref.zero_grad()
            ((a - target) ** 2).sum().backward()
            ((a_ref - target) ** 2).sum().backward()
            mine.step(); ref.step()
        assert torch.allclose(a.detach(), a_ref.detach(), atol=1e-5)
    with step("AdamW (decoupled wd) differs from Adam (L2 wd) when wd > 0"):
        torch.manual_seed(1)
        p_w = torch.randn(4, requires_grad=True)
        p_a = p_w.detach().clone().requires_grad_(True)
        target = torch.randn(4)
        ow = MyAdamW([p_w], lr=1e-2, weight_decay=0.1)
        oa = torch.optim.Adam([p_a], lr=1e-2, weight_decay=0.1)
        for _ in range(20):
            ow.zero_grad(); oa.zero_grad()
            ((p_w - target) ** 2).sum().backward()
            ((p_a - target) ** 2).sum().backward()
            ow.step(); oa.step()
        assert not torch.allclose(p_w.detach(), p_a.detach(), atol=1e-3)

