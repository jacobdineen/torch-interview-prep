"""
Step 0003: set_pad_token_to_eos

Part 1 — Model Setup and Decoding Strategies
GPT-2 has no pad token; reuse the EOS token for padding. Returns the tokenizer.
"""
import torch  # noqa: F401


def set_pad_token_to_eos(tokenizer):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
