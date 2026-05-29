"""
Step 0063: logits_to_probs_rowwise

Part 4 — Single-Layer Neural Bigram
Row-wise softmax turning logits into next-token probabilities.
"""
import numpy as np  # noqa: F401


def logits_to_probs_rowwise(logits):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
