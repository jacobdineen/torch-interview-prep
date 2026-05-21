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
from p19a_cross_entropy import *

def test_p19a_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(7, 5) * 3
    targets = torch.randint(0, 5, (7,))
    for red in ('mean', 'sum', 'none'):
        with step(f'cross_entropy matches F.cross_entropy (reduction={red!r})'):
            expected = F.cross_entropy(logits, targets, reduction=red)
            got = cross_entropy(logits, targets, reduction=red)
            assert torch.allclose(got, expected, atol=1e-06)
    pass
    pass
