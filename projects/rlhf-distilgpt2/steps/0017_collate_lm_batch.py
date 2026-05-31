"""
Step 0017: collate_lm_batch

Part 2 — SFT Data Pipeline
Pad a batch of tokenized examples into input_ids / labels / attention_mask
tensors, masking prompt tokens and padding in the labels.
"""
import torch  # noqa: F401


def collate_lm_batch(tokenized_examples, pad_id, ignore_index=-100):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
