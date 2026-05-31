"""
Step 0036: run_mcts

Part 4 — PUCT Monte Carlo Tree Search
Run ``n_simulations`` simulations from ``root`` and return it.
"""
import torch  # noqa: F401


def run_mcts(root, net, n_simulations, c_puct):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
