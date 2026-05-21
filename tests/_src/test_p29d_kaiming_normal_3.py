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
from p29d_kaiming_normal_3 import *

def test_p29d_kaiming_normal_3():
    torch.manual_seed(0)
    pass
    pass
    pass
    with step('kaiming_normal_ returns the tensor'):
        out = kaiming_normal_(torch.empty(8, 8))
        assert isinstance(out, torch.Tensor)
