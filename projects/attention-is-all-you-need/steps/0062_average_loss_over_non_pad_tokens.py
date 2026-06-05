"""
Step 0062: average_loss_over_non_pad_tokens

Part 8 — Training Objective and Schedule
Normalize the summed loss by the number of non-pad target tokens.
"""
import torch  # noqa: F401


def average_loss_over_non_pad_tokens(total_loss, n_tokens):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
