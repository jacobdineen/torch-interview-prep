"""
Step 0030: init_lora_weights

Part 4 — LoRA Adapters
LoRA init: A ~ small Gaussian (r, in), B = zeros (out, r) so the initial
delta is exactly zero. Returns (A, B).
"""
import torch  # noqa: F401


def init_lora_weights(in_features, out_features, r, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
