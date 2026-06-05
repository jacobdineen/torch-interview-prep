"""
Step 0056: shift_targets_right_with_start_token

Part 8 — Training Objective and Schedule
Build the decoder input by prepending a start token column and dropping the final target position.
"""
import torch  # noqa: F401


def shift_targets_right_with_start_token(tgt_ids, bos_id=1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
