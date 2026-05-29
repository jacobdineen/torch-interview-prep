"""
Step 0134: ffn_backward

Part 7 — FFN, Blocks, and Full Model
Backward through the 2-layer ReLU FFN. Returns (dx, dW1, db1, dW2, db2).
"""
import numpy as np  # noqa: F401


def ffn_backward(dout, x, w1, b1, w2, b2):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
