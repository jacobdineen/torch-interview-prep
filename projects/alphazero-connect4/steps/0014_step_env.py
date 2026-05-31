"""
Step 0014: step_env

Part 1 — Connect-4 Game Engine
Apply ``player``'s move in ``col``. Returns (next_board, reward, done) where
reward is +1 if this move wins for ``player``, else 0.
"""
import torch  # noqa: F401


def step_env(board, col, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
