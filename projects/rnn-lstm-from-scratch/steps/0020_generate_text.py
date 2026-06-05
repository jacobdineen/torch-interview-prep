"""
Step 0020: generate_text

Part 5 — Sampling & Evaluation
Autoregressively generate `length` new chars after `prompt` by sampling the model's last-step logits.
"""
import torch  # noqa: F401


def generate_text(params, stoi, itos, prompt, length, forward_fn, temperature=1.0, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
