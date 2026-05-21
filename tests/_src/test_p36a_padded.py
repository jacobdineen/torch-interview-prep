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
from p36a_padded import *

def test_p36a_padded():
    seqs = [(torch.tensor([1, 2, 3]), torch.tensor(0)), (torch.tensor([4, 5]), torch.tensor(1)), (torch.tensor([6, 7, 8, 9, 10]), torch.tensor(0))]
    padded, mask, labels = pad_collate(seqs, pad_value=-1)
    with step('padded shape (B, T_max) and mask dtype bool'):
        assert padded.shape == (3, 5)
        assert mask.shape == (3, 5) and mask.dtype == torch.bool
        assert labels.shape == (3,)
    pass
    pass
    pass
    pass
