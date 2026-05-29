"""
Step 0159: take_last_position_logits

Part 8 — Adam, Training Loop, and Generation
The logits at the final position (used to predict the next token).
"""
import numpy as np  # noqa: F401


def take_last_position_logits(logits):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
