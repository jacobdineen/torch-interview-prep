"""
Step 0063: stream_tokens

Part 8 — Evaluation and Chat Interface
Greedy generation as a generator: yield one decoded token string at a time.
"""
import torch  # noqa: F401


def stream_tokens(model, tokenizer, prompt, max_new_tokens):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
