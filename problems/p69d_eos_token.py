"""
Problem 69d: eos_token causes EOS-padded output

(Split from parent problem 69: Problem 69: Greedy Decoding With a KV Cache)
"""

import torch


def greedy_generate(step_fn, prompt_ids, max_new_tokens, eos_token=None):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for

    raise SystemExit(run_test_for(os.path.abspath(__file__)))
