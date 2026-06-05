"""
Step 0020: softmax_attention_weights

Part 3 — Masks and Scaled Dot-Product Attention
Turn attention scores into a probability distribution over keys.
"""
import torch  # noqa: F401


def softmax_attention_weights(scores):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
