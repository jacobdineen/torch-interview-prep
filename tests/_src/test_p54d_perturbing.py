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
from p54d_perturbing import *

def test_p54d_perturbing():
    pass
    pass
    torch.manual_seed(0)
    B, H, T, D = (2, 4, 6, 8)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    out = causal_attention(q, k, v)
    pass
    with step('perturbing future tokens (k, v at positions >= 4) leaves earlier outputs unchanged'):
        k2 = k.clone()
        k2[:, :, 4:] = torch.randn_like(k2[:, :, 4:])
        v2 = v.clone()
        v2[:, :, 4:] = torch.randn_like(v2[:, :, 4:])
        out2 = causal_attention(q, k2, v2)
        assert torch.allclose(out[:, :, :4], out2[:, :, :4], atol=1e-05)
        assert not torch.allclose(out[:, :, 4:], out2[:, :, 4:], atol=1e-05)
