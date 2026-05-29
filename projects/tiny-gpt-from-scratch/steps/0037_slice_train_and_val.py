"""
Step 0037: slice_train_and_val

Part 3 — Data Pipeline and Bigram Baseline
Split the token array into (train, val) at ``split_idx``.
"""
import numpy as np  # noqa: F401


def slice_train_and_val(data, split_idx):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
