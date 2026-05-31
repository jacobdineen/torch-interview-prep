"""
Step 0013: build_labels

Part 2 — SFT Data Pipeline
Labels for causal LM are a copy of the input ids (loss shifts them).
"""
import torch  # noqa: F401


def build_labels(input_ids):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
