"""
Step 0052: train_q_learning_agent

Part 3 — Tabular Q-Learning Foundations
Train a Q-learning agent (X) against a random opponent (O). Returns
(q_table, episode_rewards) where rewards are the agent's terminal rewards.
"""
import numpy as np  # noqa: F401


def train_q_learning_agent(n_episodes, alpha, gamma, epsilon, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
