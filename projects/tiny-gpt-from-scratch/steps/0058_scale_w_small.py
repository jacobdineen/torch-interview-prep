"""
Step 0058: scale_w_small

Part 4 — Single-Layer Neural Bigram
Scale the weights down so initial logits are small.
"""
import numpy as np  # noqa: F401


def scale_w_small(w, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
