"""
Step 0034: backup_value

Part 4 — PUCT Monte Carlo Tree Search
Propagate ``value`` (from the leaf player's perspective) up the path, flipping
sign at each level since players alternate.
"""
import torch  # noqa: F401


def backup_value(path, value):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
