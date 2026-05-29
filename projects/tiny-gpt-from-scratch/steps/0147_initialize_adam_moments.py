"""
Step 0147: initialize_adam_moments

Part 8 — Adam, Training Loop, and Generation
First and second moment accumulators (zeros) for one parameter array.
"""
import numpy as np  # noqa: F401


def initialize_adam_moments(param):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
