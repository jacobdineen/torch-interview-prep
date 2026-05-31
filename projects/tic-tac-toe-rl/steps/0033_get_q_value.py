"""
Step 0033: get_q_value

Part 3 — Tabular Q-Learning Foundations
Q(state, action), defaulting to 0.0 for unseen states.
"""
import numpy as np  # noqa: F401


def get_q_value(q_table, state_key, action):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
