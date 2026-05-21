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
from p10a_top_k_per_row import *

def test_p10a_top_k_per_row():
    x = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0], [9.0, 0.0, 7.0, 8.0, 6.0]])
    with step('top_k_per_row returns top-k values and indices, sorted desc'):
        vals, idx = top_k_per_row(x, k=3)
        assert vals.shape == (2, 3) and idx.shape == (2, 3)
        assert torch.equal(vals[0], torch.tensor([5.0, 4.0, 3.0]))
        assert torch.equal(idx[0], torch.tensor([1, 4, 2]))
    pass
    pass
    pass
