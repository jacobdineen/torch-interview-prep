"""
Step 0046: clipped_surrogate

Part 6 — PPO-Based RLHF
PPO clipped surrogate objective (to MAXIMIZE):
mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)).
"""
import torch  # noqa: F401


def clipped_surrogate(ratio, advantages, clip_eps):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
