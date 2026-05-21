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
import torch.nn.functional as F
from p22c_reduction import *

def test_p22c_reduction():
    torch.manual_seed(0)
    pass
    pass
    with step('reduction shape and sum=mean*N'):
        a = torch.randn(8, 16)
        p = torch.randn(8, 16)
        n = torch.randn(8, 16)
        none_out = triplet_margin_loss(a, p, n, margin=1.0, reduction='none')
        assert none_out.shape == (8,)
        sum_out = triplet_margin_loss(a, p, n, margin=1.0, reduction='sum')
        assert torch.isclose(sum_out, none_out.sum(), atol=1e-05)
    pass
    pass
