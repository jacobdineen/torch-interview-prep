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
from p46e_chain import *

def test_p46e_chain():
    pass
    pass
    pass
    pass
    with step('chain: classic ImageNet stem 224 -> 112 -> 56 -> 56'):
        layers = [{'kernel': 7, 'stride': 2, 'padding': 3}, {'kernel': 3, 'stride': 2, 'padding': 1}, {'kernel': 3, 'stride': 1, 'padding': 1}]
        assert conv_chain_shape(224, layers) == 56
    pass
    pass
