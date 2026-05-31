"""
Step 0044: policy_loss_cross_entropy

Part 6 — Losses and Training Loop
Cross-entropy between the network policy and the MCTS target distribution.
"""
import torch  # noqa: F401


def policy_loss_cross_entropy(policy_logits, target_policy):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
