"""
Problem 70a: identical target/draft: each round accepts K + 1 tokens

(Split from parent problem 70: Problem 70: Speculative Decoding (Greedy Variant))
"""
import torch

def speculative_decode(target_step, draft_step, prompt_ids, max_new_tokens, K=4):
    raise NotImplementedError
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
