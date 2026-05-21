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
import math
import torch
import torch.nn.functional as F
from p71d_matches import *

def test_p71d_matches():
    torch.manual_seed(0)
    N, D = (64, 32)
    a = torch.randn(N, D)
    pass
    pass
    pass
    with step('matches the symmetric cross-entropy reference'):
        b = torch.randn(N, D)
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        logits = a_n @ b_n.T
        labels = torch.arange(N)
        expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        got = info_nce_loss(a, b, tau=1.0)
        assert torch.allclose(got, expected, atol=1e-05)
