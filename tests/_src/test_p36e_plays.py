import contextlib


@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError with the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import contextlib
import torch
from torch.utils.data import DataLoader
from p36e_plays import *

def test_p36e_plays():
    seqs = [(torch.tensor([1, 2, 3]), torch.tensor(0)), (torch.tensor([4, 5]), torch.tensor(1)), (torch.tensor([6, 7, 8, 9, 10]), torch.tensor(0))]
    padded, mask, labels = pad_collate(seqs, pad_value=-1)
    pass
    pass
    pass
    pass
    with step('plays nicely with DataLoader'):

        class TinyDS(torch.utils.data.Dataset):

            def __len__(self):
                return 3

            def __getitem__(self, i):
                return seqs[i]
        loader = DataLoader(TinyDS(), batch_size=3, collate_fn=lambda b: pad_collate(b, pad_value=-1))
        for p, m, l in loader:
            assert p.shape == (3, 5) and m.shape == (3, 5) and (l.shape == (3,))
