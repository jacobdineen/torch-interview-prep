"""
Step 0029: lora_linear_forward

Part 4 — LoRA Adapters
Frozen base linear plus the LoRA delta: x@W^T + b + lora_delta(x).
"""
import torch  # noqa: F401


def lora_linear_forward(x, weight, bias, lora_a, lora_b, alpha, r):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
