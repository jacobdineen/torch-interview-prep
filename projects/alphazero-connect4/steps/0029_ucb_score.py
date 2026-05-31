"""
Step 0029: ucb_score

Part 4 — PUCT Monte Carlo Tree Search
PUCT score of a child from the PARENT's perspective: -Q(child) (the child's
value is from the opponent's view) + c_puct * P * sqrt(N_parent) / (1 + N_child).
"""
import torch  # noqa: F401


def ucb_score(parent_visits, child, c_puct):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
