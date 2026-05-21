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
import copy
import torch
import torch.nn as nn
from p43a_each import *

def test_p43a_each():
    torch.manual_seed(0)
    base = nn.Linear(4, 2)
    replicas = [copy.deepcopy(base) for _ in range(3)]
    xs = [torch.randn(8, 4) for _ in range(3)]
    ys = [torch.randn(8, 2) for _ in range(3)]
    for m, x, y in zip(replicas, xs, ys):
        ((m(x) - y) ** 2).mean().backward()
    expected = []
    for grads in zip(*[list((p.grad.clone() for p in r.parameters())) for r in replicas]):
        expected.append(torch.stack(grads).mean(dim=0))
    average_gradients(replicas)
    with step("each replica's grad equals the mean across replicas"):
        for params_tuple, mean_g in zip(zip(*[list(r.parameters()) for r in replicas]), expected):
            for p in params_tuple:
                assert torch.allclose(p.grad, mean_g, atol=1e-06)
    pass
