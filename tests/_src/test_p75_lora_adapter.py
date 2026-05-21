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
import torch.nn as nn
from p75_lora_adapter import *

def test_p75_lora_adapter():

    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    base_w_init = base.weight.detach().clone()
    base_b_init = base.bias.detach().clone()
    lora = LoRALinear(base, r=4, alpha=8.0)
    with step("B is zero-initialized so the adapter is identity at start"):
        assert torch.allclose(lora.lora_B, torch.zeros_like(lora.lora_B))
    x = torch.randn(3, 16)
    with step("at init, LoRA forward equals base forward"):
        assert torch.allclose(lora(x), base(x), atol=1e-6)
    with step("base parameters are frozen"):
        assert not base.weight.requires_grad
        assert not base.bias.requires_grad
    with step("trainable parameter count is r * (in + out)"):
        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        assert trainable == 4 * (16 + 8)
    with torch.no_grad():
        lora.lora_B.fill_(0.1)
    out = lora(x)
    with step("non-zero B makes output diverge from base output"):
        assert not torch.allclose(out, base(x), atol=1e-3)
    with step("LoRA formula: y = base(x) + (alpha/r) * x @ A^T @ B^T"):
        expected = base(x) + (8.0 / 4) * x @ lora.lora_A.t() @ lora.lora_B.t()
        assert torch.allclose(out, expected, atol=1e-5)
    with step("backward populates A and B grads only"):
        out.sum().backward()
        assert lora.lora_A.grad is not None and torch.isfinite(lora.lora_A.grad).all()
        assert lora.lora_B.grad is not None and torch.isfinite(lora.lora_B.grad).all()
        assert base.weight.grad is None or torch.equal(base.weight.grad, torch.zeros_like(base.weight))
    with step("merge_lora produces a Linear with combined weights"):
        merged = merge_lora(lora)
        assert isinstance(merged, nn.Linear)
        assert merged.in_features == 16 and merged.out_features == 8
        assert torch.allclose(merged(x), out, atol=1e-5)
    with step("merge_lora doesn't mutate the base layer"):
        assert torch.equal(base.weight, base_w_init)
        assert torch.equal(base.bias, base_b_init)

