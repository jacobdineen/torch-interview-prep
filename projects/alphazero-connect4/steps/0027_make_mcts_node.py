"""
Step 0027: make_mcts_node

Part 4 — PUCT Monte Carlo Tree Search
An MCTS node for a state: the board, the player to move, the prior prob of
reaching it, visit count N, value sum W, children, and an expanded flag.
"""
import torch  # noqa: F401


def make_mcts_node(board, player, prior):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
