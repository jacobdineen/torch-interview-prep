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
from p70c_partial import *

def test_p70c_partial():
    V = 8

    def make_target(rule):

        def fn(input_ids):
            B, T = input_ids.shape
            logits = torch.full((B, T, V), -1000000000.0)
            for t in range(T):
                last = int(input_ids[0, t].item())
                logits[:, t, (last + rule) % V] = 100.0
            return logits
        return fn
    pass
    pass
    with step('partial-match draft: accepts the matching prefix, then bonus'):

        def partial_draft():

            def fn(input_ids):
                B, T = input_ids.shape
                logits = torch.full((B, T, V), -1000000000.0)
                for t in range(T):
                    last = int(input_ids[0, t].item())
                    nxt = last + 1 if last in (0, 1) else last + 5
                    logits[:, t, nxt % V] = 100.0
                return logits
            return fn
        out = speculative_decode(make_target(rule=1), partial_draft(), torch.tensor([[0]]), max_new_tokens=6, K=4)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        assert torch.equal(out[:, :7], expected[:, :7])
