"""
Step 0037: pairwise_reward_loss

Part 5 — Reward Modeling
Bradley-Terry preference loss: -log sigmoid(r_chosen - r_rejected), averaged.
"""
import torch  # noqa: F401


def pairwise_reward_loss(chosen_rewards, rejected_rewards):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
