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
from p46g_transposed2 import *

def test_p46g_transposed2():
    pass
    pass
    pass
    pass
    pass
    pass
    with step('transposed conv: 28 -> 56 with k=2 s=2 p=0'):
        assert transposed_conv_out_shape(28, kernel=2, stride=2, padding=0) == 56
