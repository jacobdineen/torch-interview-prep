"""
Step 0035: run_one_simulation

Part 4 — PUCT Monte Carlo Tree Search
One MCTS simulation: select a leaf, evaluate/expand (or use the game result if
terminal), and back the value up.
"""
import torch  # noqa: F401


def run_one_simulation(root, net, c_puct):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
