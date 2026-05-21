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
from p51c_per import *

def test_p51c_per():
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
    pass
    with step('per-row outputs match running the LSTM on the unpadded sequence'):
        for i, L in enumerate(lengths.tolist()):
            out_i, (h_i, _) = lstm(seqs[i].unsqueeze(0))
            emb_i = torch.cat([h_i[0, 0], h_i[1, 0]])
            assert torch.allclose(sent[i], emb_i, atol=1e-05)
            assert torch.allclose(outputs[i, :L], out_i[0, :L], atol=1e-05)
