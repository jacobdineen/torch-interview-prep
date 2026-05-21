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
from p41a_checkpoint import *

def test_p41a_checkpoint():
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
        with step('checkpoint file exists after save'):
            assert os.path.exists(ckpt)
        model2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)
        epoch, best = load_checkpoint(ckpt, model2, opt2)
        pass
        pass
        pass
        pass
