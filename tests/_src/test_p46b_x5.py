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
from p46b_x5 import *

def test_p46b_x5():
    pass
    with step('5x5 conv with stride 2, padding 2 halves size'):
        assert conv_out_shape(32, kernel=5, stride=2, padding=2) == 16
    pass
    pass
    pass
    pass
    pass
