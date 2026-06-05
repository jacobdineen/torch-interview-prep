"""
Step 0047: stack_decoder_layers

Part 6 — Encoder, Decoder, and Full Model
Pass the input through every decoder layer in sequence and return the final hidden states.
"""
import torch  # noqa: F401


def stack_decoder_layers(x, memory, decoder_layers, tgt_mask, src_mask, n_heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
