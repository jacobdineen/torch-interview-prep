"""
Step 0051: batch_sequence_logprob

Part 6 — PPO-Based RLHF
Run the model and return the per-sequence log-prob of ``input_ids`` (B,).
"""
import torch  # noqa: F401


def batch_sequence_logprob(model, input_ids, attention_mask=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
