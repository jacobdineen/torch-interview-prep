"""
Step 0070: sgd_update_w

Part 4 — Single-Layer Neural Bigram
One SGD step: W <- W - lr * dW.
"""
import numpy as np  # noqa: F401


def sgd_update_w(w, dw, lr):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
