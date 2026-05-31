"""
Step 0020: build_policy_value_net

Part 2 — Board Encoding and Policy-Value Network
Assemble the conv backbone + policy and value heads into one module whose
forward(x) returns (policy_logits (B, n_actions), value (B,)).
"""
import torch  # noqa: F401


def build_policy_value_net(in_channels=2, hidden=32, n_actions=7):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
