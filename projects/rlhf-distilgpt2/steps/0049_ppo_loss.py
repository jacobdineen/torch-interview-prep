"""
Step 0049: ppo_loss

Part 6 — PPO-Based RLHF
Total PPO loss to MINIMIZE: -surrogate + vf_coef*value_loss - ent_coef*entropy.
"""
import torch  # noqa: F401


def ppo_loss(surrogate, value_loss, entropy, vf_coef, ent_coef):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
