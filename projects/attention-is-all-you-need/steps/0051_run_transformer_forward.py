"""
Step 0051: run_transformer_forward

Part 6 — Encoder, Decoder, and Full Model
Run the full encoder-decoder Transformer and return log-probabilities over the vocabulary.
"""
import torch  # noqa: F401


def run_transformer_forward(params, src_ids, tgt_ids):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
