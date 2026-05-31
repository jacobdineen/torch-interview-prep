"""
Step 0080: train_dqn_agent

Part 5 — Deep Q-Network Agent
Train a DQN agent (X) vs a random opponent (O). Returns the online params.
"""
import numpy as np  # noqa: F401


def train_dqn_agent(n_episodes, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
