"""
Step 0043: compute_returns

Part 6 — PPO-Based RLHF
Discounted returns-to-go G_t = sum_{k>=t} gamma^(k-t) r_k. rewards is 1-D (T,).
"""
import torch  # noqa: F401


def compute_returns(rewards, gamma):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
