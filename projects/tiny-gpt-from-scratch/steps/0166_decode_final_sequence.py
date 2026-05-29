"""
Step 0166: decode_final_sequence

Part 8 — Adam, Training Loop, and Generation
Decode the generated id sequence back to text.
"""
import numpy as np  # noqa: F401


def decode_final_sequence(ids, itos):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
