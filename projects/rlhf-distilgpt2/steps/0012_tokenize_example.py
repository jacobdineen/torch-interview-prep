"""
Step 0012: tokenize_example

Part 2 — SFT Data Pipeline
Tokenize the full text; record how many tokens belong to the prompt.
Returns {'input_ids', 'prompt_len'}.
"""
import torch  # noqa: F401


def tokenize_example(templated, tokenizer):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
