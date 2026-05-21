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
from p50e_last_real_state import *

def test_p50e_last_real_state():
    torch.manual_seed(0)
    seqs = [torch.randn(3, 5), torch.randn(1, 5), torch.randn(4, 5)]
    padded, mask, lengths = pad_and_mask(seqs)
    pass
    pass
    pass
    pass
    with step('last_real_state picks padded[b, lengths[b]-1]'):
        lr = last_real_state(padded, lengths)
        assert lr.shape == (3, 5)
        assert torch.equal(lr[0], seqs[0][-1])
        assert torch.equal(lr[1], seqs[1][-1])
        assert torch.equal(lr[2], seqs[2][-1])
