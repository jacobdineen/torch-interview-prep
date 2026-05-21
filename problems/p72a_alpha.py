"""
Problem 72a: alpha=1 equals plain CE

(Split from parent problem 72: Problem 57: Knowledge Distillation Loss)
"""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
