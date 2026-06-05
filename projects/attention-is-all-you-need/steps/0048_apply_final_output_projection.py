"""
Step 0048: apply_final_output_projection

Part 6 — Encoder, Decoder, and Full Model
Project decoder hidden states to vocabulary logits using the (vocab,D) output weight and bias.
"""
import torch  # noqa: F401


def apply_final_output_projection(h, out_weight, out_bias):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
