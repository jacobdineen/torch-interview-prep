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
from p67a_apply_repetition_penalty import *

def test_p67a_apply_repetition_penalty():
    logits = torch.tensor([[2.0, -1.0, 3.0, 0.5, -2.0]])
    gen = torch.tensor([[0, 2, 4]])
    with step('penalty=1.0 is a no-op for apply_repetition_penalty'):
        out = apply_repetition_penalty(logits, gen, penalty=1.0)
        assert torch.equal(out, logits)
    pass
    pass
    pass
    pass
    pass
