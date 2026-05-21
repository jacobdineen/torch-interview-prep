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

import math
import torch
from p33_lr_scheduler_cosine_warmup import *

def test_p33_lr_scheduler_cosine_warmup():

    with step("step=0 -> 0 (warmup start)"):
        assert math.isclose(linear_warmup_cosine_lr(0, 1.0, 10, 100), 0.0, abs_tol=1e-9)
    with step("step=warmup -> base_lr"):
        assert math.isclose(linear_warmup_cosine_lr(10, 1.0, 10, 100), 1.0, abs_tol=1e-9)
    with step("midway cosine: progress=0.5 -> base/2"):
        val = linear_warmup_cosine_lr(55, 1.0, 10, 100, min_lr=0.0)
        assert math.isclose(val, 0.5, abs_tol=1e-9)
    with step("at total_steps -> min_lr"):
        assert math.isclose(linear_warmup_cosine_lr(100, 1.0, 10, 100, min_lr=0.1), 0.1, abs_tol=1e-9)
    with step("beyond total_steps clamps to min_lr"):
        assert math.isclose(linear_warmup_cosine_lr(500, 1.0, 10, 100, min_lr=0.1), 0.1, abs_tol=1e-9)
    with step("tensor input returns tensor of same shape"):
        steps = torch.tensor([0, 5, 10, 55, 100, 500])
        lrs = linear_warmup_cosine_lr(steps, 1.0, 10, 100, min_lr=0.1)
        assert lrs.shape == steps.shape
        expected = torch.tensor([
            0.0, 0.5, 1.0,
            0.1 + 0.5 * 0.9 * (1 + math.cos(math.pi * 0.5)),
            0.1, 0.1,
        ])
        assert torch.allclose(lrs.float(), expected.float(), atol=1e-6)

