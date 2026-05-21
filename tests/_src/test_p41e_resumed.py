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
import os, tempfile
import torch
import torch.nn as nn
from p41e_resumed import *

def test_p41e_resumed():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)
    for _ in range(3):
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, 'ckpt.pt')
        save_checkpoint(ckpt, model, opt, epoch=7, best_metric=0.42)
        pass
        model2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        epoch, best = load_checkpoint(ckpt, model2, opt2)
        pass
        pass
        pass
        with step('resumed training step matches continuing the original'):
            opt.zero_grad()
            opt2.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
            ((model2(x) - y) ** 2).mean().backward()
            opt.step()
            opt2.step()
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1.detach(), p2.detach(), atol=1e-06)
