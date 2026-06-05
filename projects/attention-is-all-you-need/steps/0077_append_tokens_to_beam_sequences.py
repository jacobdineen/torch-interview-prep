"""
Step 0077: append_tokens_to_beam_sequences

Part 11 — Decoding and Beam Search
Extend each selected beam's sequence by its chosen next token.
"""
import torch  # noqa: F401


def append_tokens_to_beam_sequences(sequences, beam_indices, token_indices):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
