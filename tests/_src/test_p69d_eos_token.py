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
from p69d_eos_token import *

def test_p69d_eos_token():
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
    pass
    with step('eos_token causes EOS-padded output'):
        sf = make_step_fn()
        out2 = greedy_generate(sf, prompt, max_new_tokens=20, eos_token=EOS)
        eos_positions = (out2[0] == EOS).nonzero(as_tuple=True)[0]
        assert len(eos_positions) > 0
        first_eos = eos_positions[0].item()
        assert torch.all(out2[0, first_eos:] == EOS)
    pass
