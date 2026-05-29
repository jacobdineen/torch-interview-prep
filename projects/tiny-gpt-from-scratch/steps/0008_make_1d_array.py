"""
Step 0008: make_1d_array

Part 2 — NumPy and Softmax Foundations
Make a 1-D float array from a Python list of numbers.
"""
import numpy as np  # noqa: F401


def make_1d_array(values):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
