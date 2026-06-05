"""
Step 0019: sample_next_char

Part 5 — Sampling & Evaluation
Sample one character id from the temperature-scaled softmax of a (V,) logit vector.
"""
import torch  # noqa: F401


def sample_next_char(logits, temperature=1.0, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
