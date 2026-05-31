"""
Step 0019: train_val_split

Part 2 — SFT Data Pipeline
Split into (train, val); the last ``val_fraction`` of examples are validation.
"""
import torch  # noqa: F401


def train_val_split(data, val_fraction):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
