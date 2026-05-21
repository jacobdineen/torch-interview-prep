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
from p70a_identical import *

def test_p70a_identical():
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
    with step('identical target/draft: each round accepts K + 1 tokens'):
        f = make_target(rule=1)
        out = speculative_decode(f, f, torch.tensor([[0]]), max_new_tokens=10, K=4)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]])
        assert torch.equal(out, expected)
    pass
    pass
