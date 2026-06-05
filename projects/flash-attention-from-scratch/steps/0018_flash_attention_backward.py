"""
Step 0018: flash_attention_backward

Part 5 — Flash Backward (Recomputation)
Gradients dQ, dK, dV, recomputing the softmax from the saved log-sum-exp instead of storing it.
"""
import numpy as np  # noqa: F401


def flash_attention_backward(Q, K, V, O, dO, L, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
