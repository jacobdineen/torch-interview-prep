"""
Problem 35e: epoch_mean equals mean of all ys

(Split from parent problem 35: Problem 31: Custom Dataset and DataLoader)
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
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
