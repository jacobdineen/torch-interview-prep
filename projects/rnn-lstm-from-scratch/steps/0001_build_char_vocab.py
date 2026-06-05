"""
Step 0001: build_char_vocab

Part 1 — Character Data
Build sorted character-to-id and id-to-character mappings from text.
"""
import torch  # noqa: F401


def build_char_vocab(text):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
