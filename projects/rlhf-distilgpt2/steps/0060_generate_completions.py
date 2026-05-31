"""
Step 0060: generate_completions

Part 8 — Evaluation and Chat Interface
Greedy-generate a completion (the new text only) for each prompt.
"""
import torch  # noqa: F401


def generate_completions(model, tokenizer, prompts, max_new_tokens):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
