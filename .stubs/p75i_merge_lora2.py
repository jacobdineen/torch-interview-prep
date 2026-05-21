"""
Problem 75i: merge_lora doesn't mutate the base layer

(Split from parent problem 75: Problem 75: LoRA Adapter for Linear Layers)
"""
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):

    def __init__(self, base_linear: nn.Linear, r: int, alpha: float=1.0):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

def merge_lora(lora_linear):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
