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
from p66_top_k_top_p_sampling import *

def test_p66_top_k_top_p_sampling():

    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0],
                           [5.0, 1.0, 2.0, 3.0, 4.0]])
    with step("top_k_filter keeps top-k, masks the rest to -inf"):
        f = top_k_filter(logits, k=2)
        assert torch.isinf(f[0, 0]) and f[0, 0] < 0
        assert f[0, 4] == 5.0 and f[0, 3] == 4.0
        assert f[1, 0] == 5.0 and f[1, 4] == 4.0
    with step("top_p with p=0.9 keeps the top 2 (cumprob ~0.87)"):
        f = top_p_filter(logits, p=0.9)
        assert f[0, 4] == 5.0 and f[0, 3] == 4.0
        assert torch.isinf(f[0, 2]) and f[0, 2] < 0
    with step("top_p with p=0 still keeps the argmax (always-keep rule)"):
        f = top_p_filter(logits, p=0.0)
        assert f[0, 4] == 5.0
        assert torch.isinf(f[0, 0]) and f[0, 0] < 0
    with step("sample_token returns indices within the top-k filter"):
        g = torch.Generator().manual_seed(0)
        out = sample_token(logits, temperature=1.0, top_k=2, generator=g)
        assert out.shape == (2,)
        assert out[0].item() in (3, 4)
        assert out[1].item() in (0, 4)

