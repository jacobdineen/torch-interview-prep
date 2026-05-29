"""
Step 0131: ffn_linear_one_forward

Part 7 — FFN, Blocks, and Full Model
First FFN linear layer: x @ W1 + b1 -> (..., d_ff).
"""
import numpy as np  # noqa: F401


def ffn_linear_one_forward(x, w1, b1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
