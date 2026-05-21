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
from p22d_hardest import *

def test_p22d_hardest():
    torch.manual_seed(0)
    pass
    pass
    pass
    with step('hardest-triplet: 0 loss when classes well-separated'):
        e = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]])
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        loss = hardest_triplet_loss(e, y, margin=0.5)
        assert loss.item() == 0.0
    pass
