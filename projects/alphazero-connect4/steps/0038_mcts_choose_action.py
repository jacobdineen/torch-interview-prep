"""
Step 0038: mcts_choose_action

Part 4 — PUCT Monte Carlo Tree Search
Pick a column: argmax visits at temperature 0, else sample the visit policy.
"""
import torch  # noqa: F401


def mcts_choose_action(root, temperature, rng=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
