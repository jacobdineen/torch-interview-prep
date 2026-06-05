"""
Step 0043: decoder_layer_masked_self_attention_sublayer

Part 6 — Encoder, Decoder, and Full Model
Run the decoder masked self-attention sublayer with a residual add-and-norm around it.
"""
import torch  # noqa: F401


def decoder_layer_masked_self_attention_sublayer(x, layer, tgt_mask, n_heads):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
