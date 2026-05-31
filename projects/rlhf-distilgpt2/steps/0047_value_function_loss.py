"""
Step 0047: value_function_loss

Part 6 — PPO-Based RLHF
Critic loss: 0.5 * mean((values - returns)^2).
"""
import torch  # noqa: F401


def value_function_loss(values, returns):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
