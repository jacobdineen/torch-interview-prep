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
import torch.nn as nn
from p13c_is_leaf_and_requires_grad import *

def test_p13c_is_leaf_and_requires_grad():
    torch.manual_seed(0)
    pass
    pass
    with step('is_leaf_and_requires_grad reports the right flags'):
        a = torch.randn(3, requires_grad=True)
        leaf, rg = is_leaf_and_requires_grad(a)
        assert leaf is True and rg is True
        b = a * 2
        leaf2, rg2 = is_leaf_and_requires_grad(b)
        assert leaf2 is False and rg2 is True
