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
from p22e_hardest2 import *

def test_p22e_hardest2():
    torch.manual_seed(0)
    pass
    pass
    pass
    pass
    with step('hardest-triplet: positive loss when classes overlap'):
        e2 = torch.tensor([[0.0, 0.0], [0.5, 0.0], [0.6, 0.0], [1.2, 0.0]])
        y2 = torch.tensor([0, 0, 1, 1])
        loss2 = hardest_triplet_loss(e2, y2, margin=2.0)
        assert loss2.item() > 0
