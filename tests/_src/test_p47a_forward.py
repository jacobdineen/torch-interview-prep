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
from p47a_forward import *

def test_p47a_forward():
    torch.manual_seed(0)
    net = SmallCNN(num_classes=10)
    with step('forward on (4, 1, 28, 28) produces (4, 10) logits'):
        x = torch.randn(4, 1, 28, 28)
        out = net(x)
        assert out.shape == (4, 10)
    pass
    pass
