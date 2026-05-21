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
from p29a_xavier_normal_ import *

def test_p29a_xavier_normal_():
    torch.manual_seed(0)
    with step('xavier_normal_ std ~ sqrt(2/(fan_in+fan_out))'):
        w = torch.empty(64, 128)
        xavier_normal_(w, gain=1.0)
        expected = math.sqrt(2.0 / (128 + 64))
        assert abs(w.std().item() - expected) < 0.02
    pass
    pass
    pass
