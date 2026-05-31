"""
Step 0078: compute_target_q_with_target_network

Part 5 — Deep Q-Network Agent
DQN targets: r + gamma * max_a Q_target(s') * (1 - done). Batched.
"""
import numpy as np  # noqa: F401


def compute_target_q_with_target_network(target_params, next_states, rewards, dones, gamma):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
