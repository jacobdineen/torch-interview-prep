"""
Step 0004: attention_reference

Part 1 — Standard Attention
Compute ground-truth attention: row-softmax of scaled scores applied to V.
"""
import numpy as np  # noqa: F401


def attention_reference(Q, K, V, scale):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
