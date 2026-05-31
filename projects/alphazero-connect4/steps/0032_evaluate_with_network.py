"""
Step 0032: evaluate_with_network

Part 4 — PUCT Monte Carlo Tree Search
Network evaluation of a position: (priors over 7 columns, value in [-1,1]),
from ``player``'s perspective. Illegal columns get prior 0.
"""
import torch  # noqa: F401


def evaluate_with_network(net, board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
