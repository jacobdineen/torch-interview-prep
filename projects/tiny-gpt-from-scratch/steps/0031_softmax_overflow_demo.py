"""
Step 0031: softmax_overflow_demo

Part 2 — NumPy and Softmax Foundations
Show why naive softmax overflows: run it on large logits and return the
(nan/inf-polluted) result so the test can confirm the failure mode.
"""
import numpy as np  # noqa: F401


def softmax_overflow_demo():
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
