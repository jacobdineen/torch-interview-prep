"""
Step 0060: zero_pad_column_and_pad_token_rows

Part 8 — Training Objective and Schedule
Zero out the pad-token column for every row and zero entire rows whose gold token is the pad token.
"""
import torch  # noqa: F401


def zero_pad_column_and_pad_token_rows(dist, gold, pad_id=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
