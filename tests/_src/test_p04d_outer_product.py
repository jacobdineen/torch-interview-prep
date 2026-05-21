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
from p04d_outer_product import *

def test_p04d_outer_product():
    torch.manual_seed(0)
    X = torch.randn(5, 4)
    pass
    pass
    pass
    with step('outer_product of 1D vectors'):
        u = torch.tensor([1.0, 2.0, 3.0])
        v = torch.tensor([4.0, 5.0])
        op = outer_product(u, v)
        assert op.shape == (3, 2)
        assert torch.equal(op, torch.tensor([[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]))
