"""
Step 0050: kl_penalized_reward

Part 6 — PPO-Based RLHF
Shape the reward with a per-token KL-to-reference penalty.
"""
import torch  # noqa: F401


def kl_penalized_reward(rewards, logprobs, ref_logprobs, kl_coef):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
