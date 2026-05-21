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
from p47b_parameter import *

def test_p47b_parameter():
    torch.manual_seed(0)
    net = SmallCNN(num_classes=10)
    pass
    with step('parameter count > 100k (head alone is 100k+)'):
        n = count_parameters(net)
        assert n > 100000
    pass
