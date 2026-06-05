"""
Step 0007: merge_word

Part 2 — Pair Statistics & Merging
Replace each adjacent occurrence of pair=(a,b) in the tuple with the merged symbol a+b.

Scan left to right, consuming matches non-overlapping: when symbols[i]==a and symbols[i+1]==b emit the merged symbol a+b and advance by 2, else emit symbols[i] and advance by 1. So ('a','a','a') -> ('aa','a'). Return a tuple.

Convention used throughout this tokenizer:

  The end-of-word marker is the literal string "</w>". Append it as the FINAL symbol of
  every word (so a word-final piece is distinguishable from a word-internal one), and strip
  it again when decoding back to text. Symbol sequences are tuples of strings, e.g.
  ("l", "o", "w", "</w>").
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py bpe-tokenizer-from-scratch` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def merge_word(symbols, pair):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
