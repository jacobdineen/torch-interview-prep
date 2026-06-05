"""
Step 0050: apply_log_softmax_over_vocab

Part 6 — Encoder, Decoder, and Full Model
Convert vocabulary logits into log-probabilities over the vocabulary dimension.
"""
import torch  # noqa: F401


def apply_log_softmax_over_vocab(logits):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
