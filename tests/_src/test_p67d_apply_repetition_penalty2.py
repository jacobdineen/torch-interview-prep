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
from p67d_apply_repetition_penalty2 import *

def test_p67d_apply_repetition_penalty2():
    logits = torch.tensor([[2.0, -1.0, 3.0, 0.5, -2.0]])
    gen = torch.tensor([[0, 2, 4]])
    pass
    pass
    pass
    with step('apply_repetition_penalty stacks for repeated tokens'):
        gen2 = torch.tensor([[0, 0, 2]])
        out2 = apply_repetition_penalty(logits, gen2, penalty=2.0)
        assert torch.isclose(out2[0, 0], torch.tensor(0.5))
        assert torch.isclose(out2[0, 2], torch.tensor(1.5))
    pass
    pass
