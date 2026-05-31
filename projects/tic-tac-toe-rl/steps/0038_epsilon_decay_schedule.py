"""
Step 0038: epsilon_decay_schedule

Part 3 — Tabular Q-Learning Foundations
Multiplicative epsilon decay with a floor.
"""
import numpy as np  # noqa: F401


def epsilon_decay_schedule(epsilon, decay, min_epsilon):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
