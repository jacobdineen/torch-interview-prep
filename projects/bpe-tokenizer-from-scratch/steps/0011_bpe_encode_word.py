"""
Step 0011: bpe_encode_word

Part 4 — Tokenizing & Vocab
Greedily merge the word's symbols by repeatedly applying the lowest-rank adjacent merge until none remain.
"""
import numpy as np  # noqa: F401


def bpe_encode_word(word, merge_ranks):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
