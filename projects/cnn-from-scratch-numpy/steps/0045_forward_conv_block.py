"""
Step 0045: forward_conv_block

Part 5 — Assembling LeNet
Conv -> ReLU -> MaxPool; returns (out, cache) holding the three sub-caches.
"""
import numpy as np  # noqa: F401


def forward_conv_block(x, conv_layer, pool, stride):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
