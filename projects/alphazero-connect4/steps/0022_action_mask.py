"""
Step 0022: action_mask

Part 3 — Action Masking and Policy Sampling
A length-7 float mask: 1 for legal (non-full) columns, 0 otherwise.
"""
import torch  # noqa: F401


def action_mask(board):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
