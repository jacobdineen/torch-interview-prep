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
import math
import torch
from p68b_beam2 import *

def test_p68b_beam2():
    V = 5
    EOS = 4
    table = {0: torch.log(torch.tensor([1e-06, 0.7, 0.2, 1e-06, 0.1])), 1: torch.log(torch.tensor([1e-06, 1e-06, 0.2, 0.6, 0.2])), 2: torch.log(torch.tensor([1e-06, 1e-06, 1e-06, 1e-06, 1.0])), 3: torch.log(torch.tensor([1e-06, 1e-06, 1e-06, 1e-06, 1.0]))}

    def step_fn(prefix):
        return table[int(prefix[-1].item())]
    start = torch.tensor([0])
    pass
    with step('beam=1 matches greedy decoding'):
        tokens_g, _ = beam_search(step_fn, start, beam_size=1, max_steps=10, end_token=EOS)
        assert torch.equal(tokens_g, torch.tensor([0, 1, 3, 4]))
