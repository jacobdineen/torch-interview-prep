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
from p19b_ignore_index import *

def test_p19b_ignore_index():
    torch.manual_seed(0)
    logits = torch.randn(7, 5) * 3
    targets = torch.randint(0, 5, (7,))
    for red in ('mean', 'sum', 'none'):
        pass
    with step('ignore_index excludes those positions from the mean'):
        targets2 = targets.clone()
        targets2[0] = -100
        expected = F.cross_entropy(logits, targets2, reduction='mean', ignore_index=-100)
        got = cross_entropy(logits, targets2, reduction='mean', ignore_index=-100)
        assert torch.allclose(got, expected, atol=1e-06)
    pass
