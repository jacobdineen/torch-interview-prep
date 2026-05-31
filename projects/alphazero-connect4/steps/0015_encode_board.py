"""
Step 0015: encode_board

Part 2 — Board Encoding and Policy-Value Network
Two planes from ``player``'s perspective: [my pieces, opponent pieces].
Shape (2, 6, 7), float.
"""
import torch  # noqa: F401


def encode_board(board, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
