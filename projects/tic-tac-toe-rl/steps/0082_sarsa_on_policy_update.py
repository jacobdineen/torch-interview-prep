"""
Step 0082: sarsa_on_policy_update

Part 6 — Policy Gradients & Extensions
On-policy SARSA update: uses Q(s',a') for the action actually taken next,
not the max. Q <- Q + alpha*(r + gamma*next_q - Q).
"""
import numpy as np  # noqa: F401


def sarsa_on_policy_update(q_old, alpha, reward, gamma, next_q):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
