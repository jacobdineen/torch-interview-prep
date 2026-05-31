"""
Step 0062: win_rate

Part 8 — Evaluation and Chat Interface
Fraction of items where model A's reward beats model B's.
"""
import torch  # noqa: F401


def win_rate(rewards_a, rewards_b):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
