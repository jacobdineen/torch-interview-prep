"""
Step 0035: choose_learning_rate_alpha

Part 3 — Tabular Q-Learning Foundations
A sensible tabular learning rate.
"""
import numpy as np  # noqa: F401


def choose_learning_rate_alpha():
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
