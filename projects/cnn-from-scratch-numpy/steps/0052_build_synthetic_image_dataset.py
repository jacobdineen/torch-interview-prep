"""
Step 0052: build_synthetic_image_dataset

Part 6 — Synthetic Data Pipeline
Build (X (N,1,H,W), y (N,)) of class-dependent images a CNN can learn.
"""
import numpy as np  # noqa: F401


def build_synthetic_image_dataset(n_samples, image_size, num_classes, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
