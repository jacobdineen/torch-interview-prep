"""
Step 0065: cross_entropy_loss

Part 4 — Single-Layer Neural Bigram
Mean negative log-prob of the correct tokens.
"""
import numpy as np  # noqa: F401


def cross_entropy_loss(probs, y):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
