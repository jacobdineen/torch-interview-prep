"""
Step 0044: gae_advantages

Part 6 — PPO-Based RLHF
Generalized Advantage Estimation. rewards (T,), values (T+1,) with a bootstrap
value at the end. Returns advantages (T,).
"""
import torch  # noqa: F401


def gae_advantages(rewards, values, gamma, lam):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
