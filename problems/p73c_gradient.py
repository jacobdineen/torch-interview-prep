"""
Problem 73c: gradient flows through the reparameterization

(Split from parent problem 73: Problem 56: VAE Reparameterization Trick and KL Term)
"""

import torch


def reparameterize(mu, logvar, generator=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
