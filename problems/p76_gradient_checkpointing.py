"""
Problem 76: Activation (Gradient) Checkpointing

Gradient checkpointing trades compute for memory: instead of saving activations
for the backward pass, drop them and recompute the forward pass during backward.
The standard tool in PyTorch is `torch.utils.checkpoint.checkpoint`.

Implement a small "block-checkpointed" sequential wrapper that:

  - CheckpointedSequential(blocks: list[nn.Module])
        * Stores blocks as nn.ModuleList.
        * forward(x):
            * In training mode, run each block under torch.utils.checkpoint.checkpoint
              so its activations aren't stored.
            * In eval mode (or when input doesn't require grad), call blocks normally.
        * use_reentrant=False (newer PyTorch API).

The wrapped output must be numerically identical to running blocks back-to-back,
and gradients of the input/parameters must match the non-checkpointed version.
"""

import torch
import torch.nn as nn


class CheckpointedSequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
