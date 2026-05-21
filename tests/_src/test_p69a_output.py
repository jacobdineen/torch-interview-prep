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
from p69a_output import *

def test_p69a_output():
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
    with step('output shape: prompt + max_new_tokens'):
        assert out.shape[0] == 1 and out.shape[1] == 1 + 10
    pass
    pass
    pass
    pass
