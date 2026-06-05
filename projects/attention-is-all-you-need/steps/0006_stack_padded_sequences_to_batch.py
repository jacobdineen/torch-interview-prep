"""
Step 0006: stack_padded_sequences_to_batch

Part 1 — Tokenization and Batching
Pad every id sequence to the batch's max length and stack into a (B, Smax) LongTensor.
"""
import torch  # noqa: F401


def stack_padded_sequences_to_batch(list_of_id_lists, pad_id=0):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
