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
from p38a_accumulated import *

def test_p38a_accumulated():
    torch.manual_seed(0)

    def make_model_and_opt():
        m = nn.Linear(4, 2)
        return (m, torch.optim.SGD(m.parameters(), lr=0.1))

    def loss_fn(model, x, y):
        return ((model(x) - y) ** 2).mean()
    K = 4
    micro = [(torch.randn(8, 4), torch.randn(8, 2)) for _ in range(K)]
    ref_model, ref_opt = make_model_and_opt()
    big_x = torch.cat([b[0] for b in micro])
    big_y = torch.cat([b[1] for b in micro])
    ref_opt.zero_grad()
    loss_fn(ref_model, big_x, big_y).backward()
    ref_grads = [p.grad.clone() for p in ref_model.parameters()]
    accum_model, accum_opt = make_model_and_opt()
    with torch.no_grad():
        for p_a, p_r in zip(accum_model.parameters(), ref_model.parameters()):
            p_a.copy_(p_r)
    total = train_step_with_accumulation(accum_model, accum_opt, micro, loss_fn, K)
    with step('accumulated gradient equals mean-over-full-batch gradient'):
        for p, g_ref in zip(accum_model.parameters(), ref_grads):
            assert torch.allclose(p.grad, g_ref, atol=1e-05)
    pass
