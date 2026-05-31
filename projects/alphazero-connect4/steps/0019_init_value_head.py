"""
Step 0019: init_value_head

Part 2 — Board Encoding and Policy-Value Network
Value head -> a scalar in [-1, 1] via tanh.
"""
import torch  # noqa: F401


def init_value_head(input_dim):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
