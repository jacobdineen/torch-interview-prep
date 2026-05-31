"""
Step 0016: make_attention_mask

Part 2 — SFT Data Pipeline
1 for real tokens, 0 for padding.
"""
import torch  # noqa: F401


def make_attention_mask(input_ids, pad_id):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
