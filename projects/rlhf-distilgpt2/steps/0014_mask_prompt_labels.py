"""
Step 0014: mask_prompt_labels

Part 2 — SFT Data Pipeline
Set the prompt-token labels to ignore_index so loss only covers the response.
"""
import torch  # noqa: F401


def mask_prompt_labels(labels, prompt_len, ignore_index=-100):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
