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
from p36c_longest import *

def test_p36c_longest():
    seqs = [(torch.tensor([1, 2, 3]), torch.tensor(0)), (torch.tensor([4, 5]), torch.tensor(1)), (torch.tensor([6, 7, 8, 9, 10]), torch.tensor(0))]
    padded, mask, labels = pad_collate(seqs, pad_value=-1)
    pass
    pass
    with step('longest sequence has full mask True'):
        assert torch.equal(padded[2], torch.tensor([6, 7, 8, 9, 10]))
        assert mask[2].all()
    pass
    pass
