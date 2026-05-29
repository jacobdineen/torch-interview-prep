"""
Step 0111: attention_value_backward

Part 6 — Embeddings and Self-Attention
Backward of attn_out = weights @ V: (dweights, dV).
"""
import numpy as np  # noqa: F401


def attention_value_backward(dattn_out, weights, v):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
