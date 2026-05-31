"""
Step 0054: self_play_episode

Part 4 — Self-Play, Evaluation & Persistence
Run one self-play training episode (the agent plays both sides, learning
from each via perspective flipping). Mutates and returns (q_table, status).
"""
import numpy as np  # noqa: F401


def self_play_episode(q_table, alpha, gamma, epsilon, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
