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
import torch.nn.functional as F
from p20_label_smoothing_ce import *

def test_p20_label_smoothing_ce():

    torch.manual_seed(0)
    logits = torch.randn(8, 5); targets = torch.randint(0, 5, (8,))
    with step("smoothing=0 equals plain cross-entropy"):
        expected = F.cross_entropy(logits, targets)
        got = label_smoothing_ce(logits, targets, smoothing=0.0)
        assert torch.allclose(got, expected, atol=1e-6)
    with step("smoothing=0.1 matches F.cross_entropy(label_smoothing=0.1)"):
        expected = F.cross_entropy(logits, targets, label_smoothing=0.1)
        got = label_smoothing_ce(logits, targets, smoothing=0.1)
        assert torch.allclose(got, expected, atol=1e-6)
    with step("smoothing=1.0 equals -mean(log_softmax)"):
        log_probs = F.log_softmax(logits, dim=-1)
        expected = -log_probs.mean(dim=-1).mean()
        got = label_smoothing_ce(logits, targets, smoothing=1.0)
        assert torch.allclose(got, expected, atol=1e-6)
    with step("reduction='none' returns per-sample loss"):
        got_none = label_smoothing_ce(logits, targets, smoothing=0.1, reduction="none")
        assert got_none.shape == (8,)
        got_sum = label_smoothing_ce(logits, targets, smoothing=0.1, reduction="sum")
        assert torch.isclose(got_sum, got_none.sum(), atol=1e-6)

