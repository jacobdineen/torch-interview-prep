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
from p69c_subsequent import *

def test_p69c_subsequent():
    V = 8
    EOS = 7

    def make_step_fn():

        def step_fn(input_ids, cache):
            B, Tn = input_ids.shape
            past = 0 if cache is None else int(cache['past'])
            logits = torch.full((B, Tn, V), -1000000000.0)
            for t in range(Tn):
                target = (past + t + 1) % V
                logits[:, t, target] = 100.0
            return (logits, {'past': past + Tn})
        return step_fn
    step_fn = make_step_fn()
    prompt = torch.tensor([[0]])
    out = greedy_generate(step_fn, prompt, max_new_tokens=10)
    pass
    pass
    with step('subsequent tokens follow the deterministic pattern'):
        for i in range(1, 8):
            assert out[0, 1 + i].item() == (1 + i) % V
    pass
    pass
