"""
Step 0035: format_preference

Part 5 — Reward Modeling
Build the chosen/rejected full texts from a preference example.
"""
import torch  # noqa: F401


def format_preference(example):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
