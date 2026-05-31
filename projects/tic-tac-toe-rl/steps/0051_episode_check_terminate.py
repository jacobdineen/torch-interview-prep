"""
Step 0051: episode_check_terminate

Part 3 — Tabular Q-Learning Foundations
Whether the episode has ended.
"""
import numpy as np  # noqa: F401


def episode_check_terminate(status):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
