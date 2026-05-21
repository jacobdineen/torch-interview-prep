"""
Problem 42a: ema_update: ema = 0.9 * 0 + 0.1 * model

(Split from parent problem 42: Problem 59: Exponential Moving Average (EMA) of Weights)
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
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
