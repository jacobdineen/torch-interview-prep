"""
Problem 08: Gather and Scatter

  - gather_per_row(x, idx)
        x:   (N, K)
        idx: (N,) long; idx[n] in [0, K)
        return (N,) where out[n] = x[n, idx[n]]
        Use torch.gather (or equivalent advanced indexing).

  - one_hot(labels, num_classes)
        labels: (N,) long
        return (N, num_classes) float one-hot. Implement with scatter.

  - scatter_sum_rows(values, index, num_rows)
        values: (M, D)
        index : (M,)  long, index[m] in [0, num_rows)
        return (num_rows, D) where row r = sum of values[m] for m with index[m]==r
        (i.e., segment sum)
"""

import torch

def gather_per_row(x, idx):
    raise NotImplementedError

def one_hot(labels, num_classes):
    raise NotImplementedError

def scatter_sum_rows(values, index, num_rows):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
