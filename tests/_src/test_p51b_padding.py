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
from p51b_padding import *

def test_p51b_padding():
    torch.manual_seed(0)
    D, H = (4, 6)
    lstm = nn.LSTM(D, H, bidirectional=True, batch_first=True)
    padded = torch.zeros(3, 7, D)
    seqs = [torch.randn(3, D), torch.randn(7, D), torch.randn(5, D)]
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    lengths = torch.tensor([3, 7, 5])
    outputs, sent = encode_packed(lstm, padded, lengths)
    pass
    with step('padding positions in outputs are zero'):
        for i, L in enumerate(lengths.tolist()):
            if L < 7:
                assert torch.all(outputs[i, L:] == 0)
    pass
