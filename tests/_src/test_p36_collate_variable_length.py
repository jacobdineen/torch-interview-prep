import contextlib

@contextlib.contextmanager
def step(label):
    """Wrap an assertion / sub-check; on failure, raise AssertionError including the label."""
    try:
        yield
    except AssertionError as e:
        msg = str(e) if str(e) else "(no message)"
        raise AssertionError(f"step {label!r}: {msg}") from e
    except Exception as e:
        raise AssertionError(f"step {label!r} crashed with {type(e).__name__}: {e}") from e

import torch
from torch.utils.data import DataLoader
from p36_collate_variable_length import *

def test_p36_collate_variable_length():

    seqs = [
        (torch.tensor([1, 2, 3]),       torch.tensor(0)),
        (torch.tensor([4, 5]),          torch.tensor(1)),
        (torch.tensor([6, 7, 8, 9, 10]), torch.tensor(0)),
    ]
    padded, mask, labels = pad_collate(seqs, pad_value=-1)
    with step("padded shape (B, T_max) and mask dtype bool"):
        assert padded.shape == (3, 5)
        assert mask.shape == (3, 5) and mask.dtype == torch.bool
        assert labels.shape == (3,)
    with step("short sequences padded correctly with mask False"):
        assert torch.equal(padded[1], torch.tensor([4, 5, -1, -1, -1]))
        assert torch.equal(mask[1], torch.tensor([True, True, False, False, False]))
    with step("longest sequence has full mask True"):
        assert torch.equal(padded[2], torch.tensor([6, 7, 8, 9, 10]))
        assert mask[2].all()
    with step("labels preserved in batch order"):
        assert torch.equal(labels, torch.tensor([0, 1, 0]))
    with step("plays nicely with DataLoader"):
        class TinyDS(torch.utils.data.Dataset):
            def __len__(self): return 3
            def __getitem__(self, i): return seqs[i]
        loader = DataLoader(TinyDS(), batch_size=3,
                            collate_fn=lambda b: pad_collate(b, pad_value=-1))
        for p, m, l in loader:
            assert p.shape == (3, 5) and m.shape == (3, 5) and l.shape == (3,)

