"""
Problem 75: LoRA Adapter for Linear Layers

Low-Rank Adaptation: instead of fine-tuning the full weight W (d_out, d_in), add a
low-rank update B @ A where A is (r, d_in) and B is (d_out, r) for small r. The
base W is frozen.

    y = x @ W^T + (alpha / r) * x @ A^T @ B^T

with A initialized ~ N(0, std^2) (LLaMA-PEFT uses Kaiming for A) and B initialized
to zero so the adapter starts as identity.

Implement:

  - LoRALinear(base_linear: nn.Linear, r: int, alpha: float = 1.0)
        * `base_linear` weights are frozen (requires_grad=False).
        * Adds Parameters `lora_A` (r, in_features) and `lora_B` (out_features, r).
        * Initializes A with kaiming_uniform_ and B with zeros.
        * forward(x): y_base = base_linear(x); y_lora = (alpha/r) * x @ A^T @ B^T; return y_base + y_lora.

  - merge_lora(lora_linear) -> nn.Linear
        Return a NEW nn.Linear with weight = W + (alpha/r) * B @ A and the same bias.
        (Used at inference to remove the adapter overhead.)
"""

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def merge_lora(lora_linear):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
