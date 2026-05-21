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
from p50a_padded import *

def test_p50a_padded():
    torch.manual_seed(0)
    seqs = [torch.randn(3, 5), torch.randn(1, 5), torch.randn(4, 5)]
    padded, mask, lengths = pad_and_mask(seqs)
    with step('padded shape (B, T_max, D), mask bool (B, T_max), lengths correct'):
        assert padded.shape == (3, 4, 5)
        assert mask.shape == (3, 4) and mask.dtype == torch.bool
        assert torch.equal(lengths, torch.tensor([3, 1, 4]))
    pass
    pass
    pass
    pass
