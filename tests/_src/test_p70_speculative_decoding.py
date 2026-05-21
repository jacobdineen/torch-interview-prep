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
from p70_speculative_decoding import *

def test_p70_speculative_decoding():

    V = 8

    def make_target(rule):
        def fn(input_ids):
            B, T = input_ids.shape
            logits = torch.full((B, T, V), -1e9)
            for t in range(T):
                last = int(input_ids[0, t].item())
                logits[:, t, (last + rule) % V] = 100.0
            return logits
        return fn

    with step("identical target/draft: each round accepts K + 1 tokens"):
        f = make_target(rule=1)
        out = speculative_decode(f, f, torch.tensor([[0]]), max_new_tokens=10, K=4)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]])
        assert torch.equal(out, expected)
    with step("bad draft: each round accepts 0 drafts; only the target bonus token added"):
        out = speculative_decode(make_target(rule=1), make_target(rule=3),
                                  torch.tensor([[0]]), max_new_tokens=5, K=4)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5]])
        assert torch.equal(out, expected)
    with step("partial-match draft: accepts the matching prefix, then bonus"):
        def partial_draft():
            def fn(input_ids):
                B, T = input_ids.shape
                logits = torch.full((B, T, V), -1e9)
                for t in range(T):
                    last = int(input_ids[0, t].item())
                    nxt = (last + 1) if last in (0, 1) else (last + 5)
                    logits[:, t, nxt % V] = 100.0
                return logits
            return fn
        out = speculative_decode(make_target(rule=1), partial_draft(),
                                  torch.tensor([[0]]), max_new_tokens=6, K=4)
        expected = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        assert torch.equal(out[:, :7], expected[:, :7])

