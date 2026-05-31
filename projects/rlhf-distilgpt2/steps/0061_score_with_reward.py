"""
Step 0061: score_with_reward

Part 8 — Evaluation and Chat Interface
Score a list of texts with a reward model that maps input_ids -> (B,) rewards.
"""
import torch  # noqa: F401


def score_with_reward(reward_model, tokenizer, texts):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
