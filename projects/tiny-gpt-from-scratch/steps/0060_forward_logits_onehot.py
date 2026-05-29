"""
Step 0060: forward_logits_onehot

Part 4 — Single-Layer Neural Bigram
Logits as a matmul of one-hot inputs with the weights: (B,V) @ (V,V).
"""
import numpy as np  # noqa: F401


def forward_logits_onehot(onehot, w):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
