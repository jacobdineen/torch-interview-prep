"""
Step 0015: decode

Part 5 — Encode & Decode
Map ids back to tokens and reconstruct the original whitespace-joined text.

Convention used throughout this tokenizer:

  The end-of-word marker is the literal string "</w>". Append it as the FINAL symbol of
  every word (so a word-final piece is distinguishable from a word-internal one), and strip
  it again when decoding back to text. Symbol sequences are tuples of strings, e.g.
  ("l", "o", "w", "</w>").
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py bpe-tokenizer-from-scratch` (or the outline drawer) to see all signatures."""
import numpy as np  # noqa: F401


def decode(ids, id_to_token):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
