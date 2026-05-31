"""
Step 0030: select_best_child

Part 4 — PUCT Monte Carlo Tree Search
The (action, child) with the highest PUCT score.
"""
import torch  # noqa: F401


def select_best_child(node, c_puct):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
