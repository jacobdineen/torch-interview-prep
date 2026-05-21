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
from p29c_kaiming_normal_2 import *

def test_p29c_kaiming_normal_2():
    torch.manual_seed(0)
    pass
    pass
    with step('kaiming_normal_ fan_out,linear: std ~ 1/sqrt(fan_out)'):
        w3 = torch.empty(64, 128)
        kaiming_normal_(w3, mode='fan_out', nonlinearity='linear')
        expected = 1.0 / math.sqrt(64)
        assert abs(w3.std().item() - expected) < 0.02
    pass
