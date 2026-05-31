"""
Step 0068: mask_illegal_actions_neg_inf

Part 5 — Deep Q-Network Agent
Set Q-values of illegal (occupied) cells to -inf so they're never chosen.
"""
import numpy as np  # noqa: F401


def mask_illegal_actions_neg_inf(q_values, board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
