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
from p46f_transposed import *

def test_p46f_transposed():
    pass
    pass
    pass
    pass
    pass
    with step('transposed conv: 7 -> 14 with k=4 s=2 p=1'):
        assert transposed_conv_out_shape(7, kernel=4, stride=2, padding=1) == 14
    pass
