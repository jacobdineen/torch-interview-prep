"""
Step 0060: inspect_q_values_for_state

Part 4 — Self-Play, Evaluation & Persistence
The length-9 vector of Q-values for a board (zeros for unseen states).
"""
import numpy as np  # noqa: F401


def inspect_q_values_for_state(q_table, board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
