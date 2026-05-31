"""
Step 0034: build_synthetic_preference_dataset

Part 5 — Reward Modeling
Synthetic preferences: each {'prompt','chosen','rejected'} where ``chosen`` is
the better (more helpful) response.
"""
import torch  # noqa: F401


def build_synthetic_preference_dataset(n):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
