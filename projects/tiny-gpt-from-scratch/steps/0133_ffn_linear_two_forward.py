"""
Step 0133: ffn_linear_two_forward

Part 7 — FFN, Blocks, and Full Model
Second FFN linear layer: a @ W2 + b2 -> (..., d_model).
"""
import numpy as np  # noqa: F401


def ffn_linear_two_forward(a, w2, b2):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
