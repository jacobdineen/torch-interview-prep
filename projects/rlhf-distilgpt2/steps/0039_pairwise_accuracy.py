"""
Step 0039: pairwise_accuracy

Part 5 — Reward Modeling
Fraction of pairs where the chosen reward exceeds the rejected reward.
"""
import torch  # noqa: F401


def pairwise_accuracy(chosen_rewards, rejected_rewards):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
