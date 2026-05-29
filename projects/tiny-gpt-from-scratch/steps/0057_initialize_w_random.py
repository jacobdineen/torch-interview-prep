"""
Step 0057: initialize_w_random

Part 4 — Single-Layer Neural Bigram
Random (vocab, vocab) weight matrix — the learned bigram 'table'.
"""
import numpy as np  # noqa: F401


def initialize_w_random(vocab_size, rng):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
