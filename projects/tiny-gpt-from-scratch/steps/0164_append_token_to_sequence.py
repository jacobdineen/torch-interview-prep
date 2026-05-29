"""
Step 0164: append_token_to_sequence

Part 8 — Adam, Training Loop, and Generation
Append a sampled token id to the running sequence.
"""
import numpy as np  # noqa: F401


def append_token_to_sequence(ids, token):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
