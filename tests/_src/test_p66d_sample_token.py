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
from p66d_sample_token import *

def test_p66d_sample_token():
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 1.0, 2.0, 3.0, 4.0]])
    pass
    pass
    pass
    with step('sample_token returns indices within the top-k filter'):
        g = torch.Generator().manual_seed(0)
        out = sample_token(logits, temperature=1.0, top_k=2, generator=g)
        assert out.shape == (2,)
        assert out[0].item() in (3, 4)
        assert out[1].item() in (0, 4)
