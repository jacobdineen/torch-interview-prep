"""
Step 0059: set_confidence_on_gold_tokens

Part 8 — Training Objective and Schedule
Place the confidence probability mass onto each row's gold-token column of the smoothing distribution.
"""
import torch  # noqa: F401


def set_confidence_on_gold_tokens(dist, gold, confidence):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
