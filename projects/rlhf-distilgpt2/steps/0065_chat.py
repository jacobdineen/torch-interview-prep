"""
Step 0065: chat

Part 8 — Evaluation and Chat Interface
Format a user message with the instruction template, generate, and return the
model's response (without the prompt).
"""
import torch  # noqa: F401


def chat(model, tokenizer, message, max_new_tokens):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
