"""
Problem 28: Cosine Schedule With Linear Warmup

Implement a learning-rate schedule used widely in transformer training:

  - linear_warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0)

      * step in [0, warmup_steps): lr = base_lr * step / warmup_steps  (linear ramp from 0)
      * step in [warmup_steps, total_steps]:
          progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
          lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))
      * step > total_steps: clamp to min_lr.

`step` may be an int or a 1D LongTensor; if a tensor, return a tensor of the same shape.
"""

import math
import torch

def linear_warmup_cosine_lr(step, base_lr, warmup_steps, total_steps, min_lr=0.0):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
