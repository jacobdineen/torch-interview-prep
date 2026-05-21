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
from p66b_top_p import *

def test_p66b_top_p():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 1.0, 2.0, 3.0, 4.0]])
    pass
    with step('top_p with p=0.9 keeps the top 2 (cumprob ~0.87)'):
        f = top_p_filter(logits, p=0.9)
        assert f[0, 4] == 5.0 and f[0, 3] == 4.0
        assert torch.isinf(f[0, 2]) and f[0, 2] < 0
    pass
    pass
