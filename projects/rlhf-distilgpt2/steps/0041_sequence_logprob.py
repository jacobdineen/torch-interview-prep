"""
Step 0041: sequence_logprob

Part 6 — PPO-Based RLHF
Sum of the log-probs of the realized next tokens. logits (B,T,V),
input_ids (B,T) -> (B,).
"""
import torch  # noqa: F401


def sequence_logprob(logits, input_ids):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
