"""
Step 0036: reward_head_forward

Part 5 — Reward Modeling
Scalar reward per sequence from the last token's hidden state:
h_last @ w + b. hidden_states (B,T,H), weight (H,), bias scalar -> (B,).
"""
import torch  # noqa: F401


def reward_head_forward(hidden_states, weight, bias):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
