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
from p50c_real import *

def test_p50c_real():
    torch.manual_seed(0)
    seqs = [torch.randn(3, 5), torch.randn(1, 5), torch.randn(4, 5)]
    padded, mask, lengths = pad_and_mask(seqs)
    pass
    pass
    with step('real positions retain original sequence values'):
        assert torch.equal(padded[0, :3], seqs[0])
        assert torch.equal(padded[2], seqs[2])
    pass
    pass
