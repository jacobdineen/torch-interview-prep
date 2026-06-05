"""
Step 0006: best_pair

Part 2 — Pair Statistics & Merging
Return the highest-count pair, breaking ties by lexicographically smallest pair.

Return None if pair_counts is empty or None (treat a falsy argument as having no pairs).

Convention used throughout this tokenizer:

  The end-of-word marker is the literal string "</w>". Append it as the FINAL symbol of
  every word (so a word-final piece is distinguishable from a word-internal one), and strip
  it again when decoding back to text. Symbol sequences are tuples of strings, e.g.
  ("l", "o", "w", "</w>").
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py bpe-tokenizer-from-scratch` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def best_pair(pair_counts):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
