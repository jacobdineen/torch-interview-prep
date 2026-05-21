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
from p66a_top_k_filter import *

def test_p66a_top_k_filter():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 1.0, 2.0, 3.0, 4.0]])
    with step('top_k_filter keeps top-k, masks the rest to -inf'):
        f = top_k_filter(logits, k=2)
        assert torch.isinf(f[0, 0]) and f[0, 0] < 0
        assert f[0, 4] == 5.0 and f[0, 3] == 4.0
        assert f[1, 0] == 5.0 and f[1, 4] == 4.0
    pass
    pass
    pass
