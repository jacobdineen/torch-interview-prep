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
from p36b_short import *

def test_p36b_short():
    seqs = [(torch.tensor([1, 2, 3]), torch.tensor(0)), (torch.tensor([4, 5]), torch.tensor(1)), (torch.tensor([6, 7, 8, 9, 10]), torch.tensor(0))]
    padded, mask, labels = pad_collate(seqs, pad_value=-1)
    pass
    with step('short sequences padded correctly with mask False'):
        assert torch.equal(padded[1], torch.tensor([4, 5, -1, -1, -1]))
        assert torch.equal(mask[1], torch.tensor([True, True, False, False, False]))
    pass
    pass
    pass
