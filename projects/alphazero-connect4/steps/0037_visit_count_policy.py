"""
Step 0037: visit_count_policy

Part 4 — PUCT Monte Carlo Tree Search
Policy over 7 columns from child visit counts, sharpened by ``temperature``
(temperature 0 -> one-hot on the most-visited action).
"""
import torch  # noqa: F401


def visit_count_policy(root, temperature):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
