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
from p39_amp_autocast import *

def test_p39_amp_autocast():

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    x = torch.randn(4, 8)
    out = forward_autocast(model, x, dtype=torch.bfloat16, device_type="cpu")
    with step("autocast output has shape (4, 4)"):
        assert out.shape == (4, 4)
    with step("autocast output is bfloat16"):
        assert out.dtype == torch.bfloat16
    with step("autocast output is numerically close to fp32 forward"):
        out_f32 = model(x)
        assert (out.float() - out_f32).abs().max().item() < 0.5
    with step("amp_train_step reduces loss over 50 iterations"):
        opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        y = torch.randn(4, 4)
        losses = [amp_train_step(model, opt, x, y) for _ in range(50)]
        assert losses[-1] < losses[0]
    with step("master parameters remain float32 after AMP training"):
        for p in model.parameters():
            assert p.dtype == torch.float32

