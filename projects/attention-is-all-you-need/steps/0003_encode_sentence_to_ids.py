"""
Step 0003: encode_sentence_to_ids

Part 1 — Tokenization and Batching
Map a sentence's whitespace tokens to their ids, sending unknown tokens to the <unk> id 3, as a LongTensor.
"""
import torch  # noqa: F401


def encode_sentence_to_ids(sentence, token_to_id):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
