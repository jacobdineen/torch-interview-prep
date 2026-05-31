"""
Step 0050: episode_apply_q_update

Part 3 — Tabular Q-Learning Foundations
Apply one Q-learning update for (state_key, action) toward ``target``.
"""
import numpy as np  # noqa: F401


def episode_apply_q_update(q_table, state_key, action, target, alpha):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
