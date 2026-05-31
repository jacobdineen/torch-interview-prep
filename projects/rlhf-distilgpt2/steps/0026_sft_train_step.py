"""
Step 0026: sft_train_step

Part 3 — SFT Training Loop
One supervised fine-tuning step (shifted next-token loss). Returns the loss value.
"""
import torch  # noqa: F401


def sft_train_step(model, batch, optimizer):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
