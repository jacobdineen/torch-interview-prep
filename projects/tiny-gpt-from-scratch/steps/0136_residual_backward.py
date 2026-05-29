"""
Step 0136: residual_backward

Part 7 — FFN, Blocks, and Full Model
Gradient splits equally to both branches of x + sublayer: (dx, dsublayer).
"""
import numpy as np  # noqa: F401


def residual_backward(dout):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
