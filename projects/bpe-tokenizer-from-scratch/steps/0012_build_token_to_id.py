"""
Step 0012: build_token_to_id

Part 4 — Tokenizing & Vocab
Assign ids to the sorted initial symbols, then to each merged token in merge order, with no duplicates.

Convention used throughout this tokenizer:

  The end-of-word marker is the literal string "</w>". Append it as the FINAL symbol of
  every word (so a word-final piece is distinguishable from a word-internal one), and strip
  it again when decoding back to text. Symbol sequences are tuples of strings, e.g.
  ("l", "o", "w", "</w>").
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py bpe-tokenizer-from-scratch` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def build_token_to_id(initial_symbols, merges):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
