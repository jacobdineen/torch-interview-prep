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

import math
import torch
from p57_alibi_bias import *

def test_p57_alibi_bias():

    H = 4
    slopes = alibi_slopes(H)
    with step("alibi_slopes shape (num_heads,)"):
        assert slopes.shape == (H,)
    with step("alibi_slopes match m_h = 2 ** (-8 * h / H) for h=1..H"):
        expected = torch.tensor([2.0 ** (-8.0 * (i + 1) / H) for i in range(H)])
        assert torch.allclose(slopes, expected, atol=1e-6)
    T = 6
    bias = alibi_bias(H, T, causal=False)
    with step("symmetric bias shape (H, T, T)"):
        assert bias.shape == (H, T, T)
    with step("diagonal entries are zero"):
        for h in range(H):
            assert torch.allclose(bias[h].diagonal(), torch.zeros(T))
    with step("symmetric: bias[h] == bias[h].T"):
        for h in range(H):
            assert torch.allclose(bias[h], bias[h].T)
    with step("bias magnitude at offset 3: -slope * 3"):
        assert math.isclose(bias[0, 0, 3].item(), -slopes[0].item() * 3.0, abs_tol=1e-6)
    bias_c = alibi_bias(H, T, causal=True)
    with step("causal bias has shape (H, T, T) and diagonal zero"):
        assert bias_c.shape == (H, T, T)
        for h in range(H):
            assert torch.allclose(bias_c[h].diagonal(), torch.zeros(T))

