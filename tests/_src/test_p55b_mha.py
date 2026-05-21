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
from p55b_mha import *

def test_p55b_mha():
    torch.manual_seed(0)
    B, T, D, H = (2, 7, 16, 4)
    mha = MultiHeadAttention(D, H)
    x = torch.randn(B, T, D)
    out = mha(x)
    pass
    with step('MHA is permutation-equivariant under input permutation'):
        perm = torch.randperm(T)
        out_perm = mha(x[:, perm])
        assert torch.allclose(out_perm, out[:, perm], atol=1e-05)
    pass
    pass
