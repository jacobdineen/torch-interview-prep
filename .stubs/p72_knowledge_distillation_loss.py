"""
Problem 57: Knowledge Distillation Loss

Classic Hinton-style knowledge distillation: a student matches a teacher's softened
output distribution alongside the hard label cross-entropy.

  - kd_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5)
        L = alpha * CE(student_logits, targets) +
            (1 - alpha) * T^2 * KL( softmax(teacher_logits/T) || softmax(student_logits/T) )

        The T^2 factor preserves gradient magnitudes (Hinton et al., 2015).
        Use a stable KL: KL(p || q) = sum p * (log p - log q), or equivalently
        sum p * (log(softmax_T(teacher)) - log(softmax_T(student))).

        Returns a 0-D scalar (mean over the batch).
"""

import torch
import torch.nn.functional as F


def kd_loss(student_logits, teacher_logits, targets, T=4.0, alpha=0.5):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
