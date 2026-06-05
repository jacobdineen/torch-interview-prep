"""
Step 0035: softmax_cross_entropy_backward

Part 4 — Fused Loss and Optimizers
Gradient of fused softmax cross-entropy w.r.t. logits: (probs - one_hot)/N.
"""
import numpy as np  # noqa: F401


def softmax_cross_entropy_backward(cache):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
