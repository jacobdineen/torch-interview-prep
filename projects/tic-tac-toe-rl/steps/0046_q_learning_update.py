"""
Step 0046: q_learning_update

Part 3 — Tabular Q-Learning Foundations
Q-learning update: Q <- Q + alpha * (target - Q).
"""
import numpy as np  # noqa: F401


def q_learning_update(q_old, alpha, target):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
