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
import torch.nn as nn
from p51_bidirectional_lstm_packed import *

def test_p51_bidirectional_lstm_packed():

    torch.manual_seed(0)
    D, H = 4, 6
    lstm = nn.LSTM(D, H, bidirectional=True, batch_first=True)
    padded = torch.zeros(3, 7, D)
    seqs = [torch.randn(3, D), torch.randn(7, D), torch.randn(5, D)]
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
    lengths = torch.tensor([3, 7, 5])
    outputs, sent = encode_packed(lstm, padded, lengths)
    with step("outputs shape (B, T, 2H), sent shape (B, 2H)"):
        assert outputs.shape == (3, 7, 2 * H)
        assert sent.shape == (3, 2 * H)
    with step("padding positions in outputs are zero"):
        for i, L in enumerate(lengths.tolist()):
            if L < 7:
                assert torch.all(outputs[i, L:] == 0)
    with step("per-row outputs match running the LSTM on the unpadded sequence"):
        for i, L in enumerate(lengths.tolist()):
            out_i, (h_i, _) = lstm(seqs[i].unsqueeze(0))
            emb_i = torch.cat([h_i[0, 0], h_i[1, 0]])
            assert torch.allclose(sent[i], emb_i, atol=1e-5)
            assert torch.allclose(outputs[i, :L], out_i[0, :L], atol=1e-5)

