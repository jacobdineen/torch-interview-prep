"""
Problem 25: Weight Initialization

Implement Xavier (Glorot) and Kaiming (He) normal initialization in-place on a
weight tensor of shape (out_features, in_features).

  - xavier_normal_(tensor, gain=1.0)
        std = gain * sqrt(2 / (fan_in + fan_out))
        sample from N(0, std^2) in-place.

  - kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu")
        gain = sqrt(2) for relu, 1 for linear.
        std = gain / sqrt(fan)  where fan = fan_in or fan_out per `mode`.
        sample from N(0, std^2) in-place.

For 2D weight tensors, fan_in = shape[1], fan_out = shape[0].
"""

import math
import torch

def xavier_normal_(tensor, gain=1.0):
    raise NotImplementedError

def kaiming_normal_(tensor, mode="fan_in", nonlinearity="relu"):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
