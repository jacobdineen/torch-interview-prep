"""
Step 0040: reward_train_step

Part 5 — Reward Modeling
One reward-model step: score chosen/rejected, pairwise loss, update. Returns loss.
"""
import torch  # noqa: F401


def reward_train_step(reward_model, batch, optimizer):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
