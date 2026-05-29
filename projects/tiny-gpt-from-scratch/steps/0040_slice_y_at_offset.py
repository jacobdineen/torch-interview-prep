"""
Step 0040: slice_y_at_offset

Part 3 — Data Pipeline and Bigram Baseline
Target window: the inputs shifted by one, tokens [i+1, i+block_size+1).
"""
import numpy as np  # noqa: F401


def slice_y_at_offset(data, i, block_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
