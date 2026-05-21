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
from p46c_x32 import *

def test_p46c_x32():
    pass
    pass
    with step('3x3 stride-2 no-padding shrinks 7 -> 3'):
        assert conv_out_shape(7, kernel=3, stride=2, padding=0) == 3
    pass
    pass
    pass
    pass
