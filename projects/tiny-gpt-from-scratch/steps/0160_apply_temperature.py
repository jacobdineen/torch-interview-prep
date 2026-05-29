"""
Step 0160: apply_temperature

Part 8 — Adam, Training Loop, and Generation
Scale logits by 1/temperature before softmax.
"""
import numpy as np  # noqa: F401


def apply_temperature(logits, temperature):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
