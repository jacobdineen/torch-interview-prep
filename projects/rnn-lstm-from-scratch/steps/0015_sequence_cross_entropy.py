"""
Step 0015: sequence_cross_entropy

Part 4 — Loss & Training
Mean cross-entropy of (B,S,V) logits against (B,S) integer targets.
"""
import torch  # noqa: F401


def sequence_cross_entropy(logits, targets):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
