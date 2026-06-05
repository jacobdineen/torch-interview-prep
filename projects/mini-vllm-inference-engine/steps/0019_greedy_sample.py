"""
Step 0019: greedy_sample

Part 5 — Prefix Cache & Sampling
Return the index of the maximum logit.
"""
import numpy as np  # noqa: F401


def greedy_sample(logits):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
