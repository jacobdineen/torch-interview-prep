"""
Step 0058: build_uniform_smoothing_distribution

Part 8 — Training Objective and Schedule
Create the (vocab,) target distribution filled with the off-target smoothing mass spread over non-target non-pad classes.
"""
import torch  # noqa: F401


def build_uniform_smoothing_distribution(vocab_size, smoothing, pad_id=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
