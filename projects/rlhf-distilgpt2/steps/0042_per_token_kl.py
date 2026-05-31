"""
Step 0042: per_token_kl

Part 6 — PPO-Based RLHF
Per-token KL estimate between policy and reference: logp - logp_ref.
"""
import torch  # noqa: F401


def per_token_kl(logprobs, ref_logprobs):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
