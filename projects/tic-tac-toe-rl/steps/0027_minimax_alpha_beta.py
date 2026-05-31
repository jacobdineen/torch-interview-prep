"""
Step 0027: minimax_alpha_beta

Part 2 — Random and Minimax Baselines
Minimax value with alpha-beta pruning (same result as minimax_recursive).
"""
import numpy as np  # noqa: F401


def minimax_alpha_beta(board, player, alpha, beta):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
