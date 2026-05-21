import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import torch
from p69_greedy_decode_with_kv_cache import *

def test_p69_greedy_decode_with_kv_cache():

    V = 8
    EOS = 7

    def make_step_fn():
        def step_fn(input_ids, cache):
            B, Tn = input_ids.shape
            past = 0 if cache is None else int(cache["past"])
            logits = torch.full((B, Tn, V), -1e9)
            for t in range(Tn):
                target = (past + t + 1) % V
                logits[:, t, target] = 100.0
            return logits, {"past": past + Tn}
        return step_fn

    step_fn = make_step_fn()
    prompt = torch.tensor([[0]])
    out = greedy_generate(step_fn, prompt, max_new_tokens=10)
    with step("output shape: prompt + max_new_tokens"):
        assert out.shape[0] == 1 and out.shape[1] == 1 + 10
    with step("first generated token is argmax at last prompt position"):
        assert out[0, 1].item() == 1
    with step("subsequent tokens follow the deterministic pattern"):
        for i in range(1, 8):
            assert out[0, 1 + i].item() == (1 + i) % V
    with step("eos_token causes EOS-padded output"):
        sf = make_step_fn()
        out2 = greedy_generate(sf, prompt, max_new_tokens=20, eos_token=EOS)
        eos_positions = (out2[0] == EOS).nonzero(as_tuple=True)[0]
        assert len(eos_positions) > 0
        first_eos = eos_positions[0].item()
        assert torch.all(out2[0, first_eos:] == EOS)
    with step("batch mode produces independent rows with same prompts giving same output"):
        sf = make_step_fn()
        prompt_b = torch.tensor([[0], [0]])
        out_b = greedy_generate(sf, prompt_b, max_new_tokens=5)
        assert out_b.shape == (2, 6)
        assert torch.equal(out_b[0], out_b[1])

