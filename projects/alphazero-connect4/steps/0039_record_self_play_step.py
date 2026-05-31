"""
Step 0039: record_self_play_step

Part 5 — Self-Play Data Generation
Append a (state, MCTS policy, player) tuple; the value target is filled in
later from the game outcome. Returns the buffer.
"""
import torch  # noqa: F401


def record_self_play_step(buffer, state, policy, player):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
