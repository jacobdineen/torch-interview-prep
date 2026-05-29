"""
Step 0083: softmax_cross_entropy_backward

Part 5 — Layer Primitives and Backprop
dL/dlogits for mean softmax+cross-entropy over a batch: (probs - onehot)/N.
"""
import numpy as np  # noqa: F401


def softmax_cross_entropy_backward(probs, y):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
