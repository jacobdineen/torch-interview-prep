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
from p46d_dilation import *

def test_p46d_dilation():
    pass
    pass
    pass
    with step('dilation=2 effective kernel: 7 -> 2'):
        assert conv_out_shape(7, kernel=3, stride=2, padding=0, dilation=2) == 2
    pass
    pass
    pass
