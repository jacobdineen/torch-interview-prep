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
from p50_pack_pad_sequences import *

def test_p50_pack_pad_sequences():

    torch.manual_seed(0)
    seqs = [torch.randn(3, 5), torch.randn(1, 5), torch.randn(4, 5)]
    padded, mask, lengths = pad_and_mask(seqs)
    with step("padded shape (B, T_max, D), mask bool (B, T_max), lengths correct"):
        assert padded.shape == (3, 4, 5)
        assert mask.shape == (3, 4) and mask.dtype == torch.bool
        assert torch.equal(lengths, torch.tensor([3, 1, 4]))
    with step("padded positions are zeros"):
        assert torch.equal(padded[0, 3], torch.zeros(5))
        assert torch.equal(padded[1, 1:], torch.zeros(3, 5))
    with step("real positions retain original sequence values"):
        assert torch.equal(padded[0, :3], seqs[0])
        assert torch.equal(padded[2], seqs[2])
    with step("masked_mean_pool averages over real positions per sequence"):
        mp = masked_mean_pool(padded, mask)
        assert mp.shape == (3, 5)
        assert torch.allclose(mp[0], seqs[0].mean(dim=0), atol=1e-6)
        assert torch.allclose(mp[1], seqs[1].mean(dim=0), atol=1e-6)
    with step("last_real_state picks padded[b, lengths[b]-1]"):
        lr = last_real_state(padded, lengths)
        assert lr.shape == (3, 5)
        assert torch.equal(lr[0], seqs[0][-1])
        assert torch.equal(lr[1], seqs[1][-1])
        assert torch.equal(lr[2], seqs[2][-1])

