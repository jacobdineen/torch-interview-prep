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
from p74d_gumbel_softmax import *

def test_p74d_gumbel_softmax():
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    samples = gumbel_noise((100000,), generator=g)
    pass
    pass
    pass
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    with step('soft gumbel_softmax produces probability simplex (rows sum to 1)'):
        g2 = torch.Generator().manual_seed(0)
        y = gumbel_softmax(logits, tau=1.0, hard=False, generator=g2)
        assert y.shape == logits.shape
        assert torch.allclose(y.sum(dim=-1), torch.ones(1), atol=1e-05)
        assert (y >= 0).all()
    pass
    pass
