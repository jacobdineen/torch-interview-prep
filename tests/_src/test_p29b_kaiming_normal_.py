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
from p29b_kaiming_normal_ import *

def test_p29b_kaiming_normal_():
    torch.manual_seed(0)
    pass
    with step('kaiming_normal_ fan_in,relu: std ~ sqrt(2)/sqrt(fan_in)'):
        w2 = torch.empty(64, 128)
        kaiming_normal_(w2, mode='fan_in', nonlinearity='relu')
        expected = math.sqrt(2.0) / math.sqrt(128)
        assert abs(w2.std().item() - expected) < 0.02
    pass
    pass
