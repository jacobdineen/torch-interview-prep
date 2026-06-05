"""
Step 0044: init_lenet

Part 5 — Assembling LeNet
Build a small LeNet: 2 conv blocks (3x3 same-pad, pool2) then 2 FC + logits layer.
"""
import numpy as np  # noqa: F401


def init_lenet(in_ch, num_classes, image_size, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
