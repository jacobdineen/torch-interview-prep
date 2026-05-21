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
from p16_softmax_logsumexp import *

def test_p16_softmax_logsumexp():

    torch.manual_seed(0)
    x = torch.randn(4, 5)
    with step("logsumexp matches torch.logsumexp"):
        assert torch.allclose(logsumexp(x, dim=1), torch.logsumexp(x, dim=1), atol=1e-6)
    with step("softmax matches torch.softmax"):
        assert torch.allclose(softmax(x, dim=1), torch.softmax(x, dim=1), atol=1e-6)
    with step("log_softmax matches torch.log_softmax"):
        assert torch.allclose(log_softmax(x, dim=-1), torch.log_softmax(x, dim=-1), atol=1e-6)
    with step("softmax rows sum to 1"):
        s = softmax(x, dim=1)
        assert torch.allclose(s.sum(dim=1), torch.ones(4), atol=1e-6)
    with step("softmax stable at logits of order 1e4"):
        big = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 1.0]])
        sb = softmax(big, dim=1)
        assert torch.isfinite(sb).all()
        assert torch.allclose(sb.sum(dim=1), torch.ones(1), atol=1e-6)
    with step("logsumexp stable at logits of order 1e4"):
        big = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 1.0]])
        lse = logsumexp(big, dim=1)
        assert torch.isfinite(lse).all()

