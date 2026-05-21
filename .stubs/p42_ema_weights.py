"""
Problem 59: Exponential Moving Average (EMA) of Weights

EMA model weights are widely used in self-supervised pre-training (BYOL, MoCo) and
diffusion models. The EMA model is updated each step with:

    ema_param = decay * ema_param + (1 - decay) * model_param

Implement:

  - ema_update(ema_model, model, decay=0.999)
        In-place update of ema_model.parameters() and buffers based on model.
        Buffers are typically just copied (not averaged) — running stats from BN, etc.

  - EMAWrapper(model, decay=0.999)
        Class wrapping a model + its EMA copy. Methods:
            update(): call ema_update with the wrapped model.
            ema_state_dict(): return EMA model's state_dict.
        The EMA copy must be deep-copied at construction and have requires_grad=False
        on its parameters.
"""

import copy
import torch
import torch.nn as nn


def ema_update(ema_model, model, decay=0.999):
    raise NotImplementedError


class EMAWrapper:
    def __init__(self, model, decay=0.999):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def ema_state_dict(self):
        raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
