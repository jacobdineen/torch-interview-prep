"""
Problem 23: Inverted Dropout From Scratch

Implement inverted dropout (the standard form). At train time, drop each element
independently with probability p and scale survivors by 1/(1-p). At eval time, act
as identity. Use the provided generator for reproducibility.

  - MyDropout(p=0.5)
        forward(x): in training, x_dropped = x * mask / (1 - p) where mask ~ Bernoulli(1-p).
                    in eval, returns x unchanged.
"""

import torch
import torch.nn as nn

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
