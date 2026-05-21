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

import os, tempfile
import torch
import torch.nn as nn
from p41_checkpoint_save_load import *

def test_p41_checkpoint_save_load():

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(16, 4); y = torch.randn(16, 2)
    for _ in range(3):
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "ckpt.pt")
        save_checkpoint(ckpt, model, opt, epoch=7, best_metric=0.42)
        with step("checkpoint file exists after save"):
            assert os.path.exists(ckpt)
        model2 = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        epoch, best = load_checkpoint(ckpt, model2, opt2)
        with step("epoch and best_metric restored"):
            assert epoch == 7
            assert abs(best - 0.42) < 1e-9
        with step("model parameters restored bit-exact"):
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.equal(p1.detach(), p2.detach())
        with step("optimizer state (Adam moments) restored bit-exact"):
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                s1 = opt.state[p1]; s2 = opt2.state[p2]
                if "exp_avg" in s1:
                    assert torch.equal(s1["exp_avg"], s2["exp_avg"])
                    assert torch.equal(s1["exp_avg_sq"], s2["exp_avg_sq"])
        with step("resumed training step matches continuing the original"):
            opt.zero_grad(); opt2.zero_grad()
            ((model(x) - y) ** 2).mean().backward()
            ((model2(x) - y) ** 2).mean().backward()
            opt.step(); opt2.step()
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1.detach(), p2.detach(), atol=1e-6)

