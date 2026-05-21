"""
Problem 60: Save and Load a Training Checkpoint

A real training run needs to save not just model weights but also the optimizer
state, epoch number, and any best-metric tracking, so resumption is bit-exact.

  - save_checkpoint(path, model, optimizer, epoch, best_metric)
        Writes a checkpoint dict to `path` using torch.save with keys:
            "model": model.state_dict()
            "optimizer": optimizer.state_dict()
            "epoch": int
            "best_metric": float

  - load_checkpoint(path, model, optimizer)
        Loads the checkpoint at `path` and restores model + optimizer state in place.
        Returns (epoch, best_metric).
"""

import torch


def save_checkpoint(path, model, optimizer, epoch, best_metric):
    raise NotImplementedError


def load_checkpoint(path, model, optimizer):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
