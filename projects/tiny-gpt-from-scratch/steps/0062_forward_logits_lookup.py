"""
Step 0062: forward_logits_lookup

Part 4 — Single-Layer Neural Bigram
Logits via row lookup W[x] — same result as one-hot @ W, far cheaper.
"""
import numpy as np  # noqa: F401


def forward_logits_lookup(x, w):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
