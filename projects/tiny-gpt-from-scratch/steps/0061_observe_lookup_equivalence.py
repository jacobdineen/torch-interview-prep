"""
Step 0061: observe_lookup_equivalence

Part 4 — Single-Layer Neural Bigram
One-hot @ W equals a plain row lookup W[x]; return True to confirm.
"""
import numpy as np  # noqa: F401


def observe_lookup_equivalence(x, w):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
