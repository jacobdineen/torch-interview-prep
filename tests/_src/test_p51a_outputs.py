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
import torch.nn as nn
from p51a_outputs import *

def test_p51a_outputs():
    torch.manual_seed(0)
    D, H = (4, 6)
    lstm = nn.LSTM(D, H, bidirectional=True, batch_first=True)
    padded = torch.zeros(3, 7, D)
    seqs = [torch.randn(3, D), torch.randn(7, D), torch.randn(5, D)]
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    lengths = torch.tensor([3, 7, 5])
    outputs, sent = encode_packed(lstm, padded, lengths)
    with step('outputs shape (B, T, 2H), sent shape (B, 2H)'):
        assert outputs.shape == (3, 7, 2 * H)
        assert sent.shape == (3, 2 * H)
    pass
    pass
