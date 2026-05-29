"""
Step 0161: top_k_filter

Part 8 — Adam, Training Loop, and Generation
Keep the top-k logits, set the rest to a large negative number.
"""
import numpy as np  # noqa: F401


def top_k_filter(logits, k):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
