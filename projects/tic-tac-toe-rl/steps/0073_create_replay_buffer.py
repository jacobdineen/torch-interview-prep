"""
Step 0073: create_replay_buffer

Part 5 — Deep Q-Network Agent
An empty replay buffer (list of transitions).
"""
import numpy as np  # noqa: F401


def create_replay_buffer():
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
