"""
Step 0021: policy_value_forward

Part 2 — Board Encoding and Policy-Value Network
Run the network: returns (policy_logits, value).
"""
import torch  # noqa: F401


def policy_value_forward(net, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
