"""
Step 0070: mse_loss_on_chosen_action

Part 5 — Deep Q-Network Agent
Mean squared error on the taken action only (DQN updates that action).
q_pred (batch,9), actions (batch,), targets (batch,).
"""
import numpy as np  # noqa: F401


def mse_loss_on_chosen_action(q_pred, actions, targets):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
