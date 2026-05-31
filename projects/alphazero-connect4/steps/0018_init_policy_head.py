"""
Step 0018: init_policy_head

Part 2 — Board Encoding and Policy-Value Network
Linear policy head over the flattened features -> action logits.
"""
import torch  # noqa: F401


def init_policy_head(input_dim, n_actions):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
