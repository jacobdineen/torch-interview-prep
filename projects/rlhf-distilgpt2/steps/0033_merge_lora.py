"""
Step 0033: merge_lora

Part 4 — LoRA Adapters
Fold the LoRA update into the base weight: W + (alpha/r) * (B @ A). (out,in).
"""
import torch  # noqa: F401


def merge_lora(weight, lora_a, lora_b, alpha, r):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
