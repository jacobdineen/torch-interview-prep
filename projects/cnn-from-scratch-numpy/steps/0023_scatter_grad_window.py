"""
Step 0023: scatter_grad_window

Part 3 — Layer Forward and Backward Passes
Route gradient to argmax position of a pooling window (1 at max else 0, times dout_val).
"""
import numpy as np  # noqa: F401


def scatter_grad_window(dout_val, window):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
