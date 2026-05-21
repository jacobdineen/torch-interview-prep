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
import torch.nn.functional as F
from p64b_logits2 import *

def test_p64b_logits2():
    torch.manual_seed(0)
    V, D = (100, 8)
    lm = TiedLM(V, D)
    ids = torch.randint(0, V, (3, 5))
    out = lm(ids)
    pass
    with step('logits equal embed(ids) @ embed.weight.T'):
        expected = lm.embed(ids) @ lm.embed.weight.t()
        assert torch.allclose(out, expected, atol=1e-06)
    pass
    pass
    pass
