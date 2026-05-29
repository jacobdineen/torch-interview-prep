"""
Step 0156: encode_prompt

Part 8 — Adam, Training Loop, and Generation
Encode a prompt string into a 1-D int array of token ids.
"""
import numpy as np  # noqa: F401


def encode_prompt(s, stoi):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
