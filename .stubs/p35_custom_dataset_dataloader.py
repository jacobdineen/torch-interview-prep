"""
Problem 31: Custom Dataset and DataLoader

Implement a map-style Dataset and use a DataLoader with shuffling.

  - SyntheticDataset(num_samples, dim, seed=0)
        * __len__ returns num_samples
        * __getitem__(i) returns (x_i, y_i) where x_i is a (dim,) Tensor and y_i is
          a 0-D Tensor: y_i = sum(x_i ** 2)
        * Use a Generator seeded with `seed` so the dataset is deterministic across
          processes / workers.

  - make_loader(dataset, batch_size, shuffle): returns a torch.utils.data.DataLoader.

  - epoch_mean(loader): iterate `loader` once, return a 0-D Tensor with mean of all y's.
"""

import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, dim, seed=0):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

def make_loader(dataset, batch_size, shuffle):
    raise NotImplementedError

def epoch_mean(loader):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
