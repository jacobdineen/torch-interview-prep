"""
Step 0049: tie_output_projection_to_token_embeddings

Part 6 — Encoder, Decoder, and Full Model
Return the token embedding matrix to be reused as the tied output projection weight.
"""
import torch  # noqa: F401


def tie_output_projection_to_token_embeddings(embed):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
