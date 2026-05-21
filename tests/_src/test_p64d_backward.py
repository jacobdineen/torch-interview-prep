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
from p64d_backward import *

def test_p64d_backward():
    torch.manual_seed(0)
    V, D = (100, 8)
    lm = TiedLM(V, D)
    ids = torch.randint(0, V, (3, 5))
    out = lm(ids)
    pass
    pass
    pass
    with step('backward populates embed.weight.grad'):
        targets = torch.randint(0, V, (3, 5))
        loss = F.cross_entropy(out.view(-1, V), targets.view(-1))
        loss.backward()
        assert lm.embed.weight.grad is not None
        assert torch.isfinite(lm.embed.weight.grad).all()
    pass
