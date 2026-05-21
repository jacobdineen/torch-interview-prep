"""
Problem 62: Mixed Precision With autocast

Use torch.autocast to run a forward pass in lower precision. CPU autocast supports
bfloat16; GPU autocast typically uses float16 with a GradScaler.

Implement:

  - forward_autocast(model, x, dtype=torch.bfloat16, device_type="cpu")
        Run model(x) inside a torch.autocast context. Return the output.
        The output dtype should match the autocast dtype for ops that downcast
        (linear layers, conv, etc. become low-precision; outputs are typically
        the autocast dtype unless cast back).

  - amp_train_step(model, optimizer, x, y, dtype=torch.bfloat16, device_type="cpu")
        Standard AMP step (no GradScaler since CPU bfloat16 doesn't need it):
            with autocast(...): out = model(x); loss = mse(out, y)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
        Return the loss as a Python float.

Both must run on CPU without CUDA.
"""

import torch
import torch.nn as nn


def forward_autocast(model, x, dtype=torch.bfloat16, device_type="cpu"):
    raise NotImplementedError


def amp_train_step(model, optimizer, x, y, dtype=torch.bfloat16, device_type="cpu"):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
