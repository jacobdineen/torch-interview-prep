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
from p06a_cat_features import *

def test_p06a_cat_features():
    a = torch.arange(6).reshape(3, 2)
    b = torch.arange(9).reshape(3, 3) + 100
    with step('cat_features concatenates along the feature dim'):
        c = cat_features(a, b)
        assert c.shape == (3, 5)
        assert torch.equal(c[:, :2], a) and torch.equal(c[:, 2:], b)
    pass
    pass
    pass
