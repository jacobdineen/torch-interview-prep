"""
Step 0070: compute_batch_training_loss

Part 10 — Training Step and Loop
Forward the model on shifted decoder inputs and return the label-smoothed KL loss averaged over non-pad target tokens plus that token count.
"""
import torch  # noqa: F401


def compute_batch_training_loss(params, src_ids, tgt_ids, smoothing, pad_id=0, bos_id=1):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
