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
from p22b_triplet_margin_loss import *

def test_p22b_triplet_margin_loss():
    torch.manual_seed(0)
    pass
    with step('matches F.triplet_margin_loss'):
        a = torch.randn(8, 16)
        p = torch.randn(8, 16)
        n = torch.randn(8, 16)
        ref = F.triplet_margin_loss(a, p, n, margin=1.0, p=2, reduction='mean')
        got = triplet_margin_loss(a, p, n, margin=1.0)
        assert torch.allclose(got, ref, atol=1e-05)
    pass
    pass
    pass
