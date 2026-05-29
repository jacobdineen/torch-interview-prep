"""
Step 0073: sample_from_neural_bigram

Part 4 — Single-Layer Neural Bigram
Generate ``n`` tokens by softmaxing each row W[cur] and sampling.
"""
import numpy as np  # noqa: F401


def sample_from_neural_bigram(w, start_token, n, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
