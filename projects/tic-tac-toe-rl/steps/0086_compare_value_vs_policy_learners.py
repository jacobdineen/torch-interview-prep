"""
Step 0086: compare_value_vs_policy_learners

Part 6 — Policy Gradients & Extensions
Train a value-based (tabular Q-learning) and a policy-based (tabular
REINFORCE) agent, evaluate both vs random, and return {'value','policy'} stats.
"""
import numpy as np  # noqa: F401


def compare_value_vs_policy_learners(n_episodes, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
