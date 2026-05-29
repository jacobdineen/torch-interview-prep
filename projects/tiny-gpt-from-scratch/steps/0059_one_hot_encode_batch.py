"""
Step 0059: one_hot_encode_batch

Part 4 — Single-Layer Neural Bigram
One-hot encode a batch of token ids (B,) into (B, vocab_size).
"""
import numpy as np  # noqa: F401


def one_hot_encode_batch(x, vocab_size):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
