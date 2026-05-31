"""
Step 0020: shift_logits_and_labels

Part 3 — SFT Training Loop
Align next-token prediction: drop the last logit and the first label.
(B,T,V),(B,T) -> (B,T-1,V),(B,T-1).
"""
import torch  # noqa: F401


def shift_logits_and_labels(logits, labels):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
