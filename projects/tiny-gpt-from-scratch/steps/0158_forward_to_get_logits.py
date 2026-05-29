"""
Step 0158: forward_to_get_logits

Part 8 — Adam, Training Loop, and Generation
Run the model on a single (T,) sequence; return (T, vocab) logits.
"""
import numpy as np  # noqa: F401


def forward_to_get_logits(params, ids):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
