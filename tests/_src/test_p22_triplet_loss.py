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
from p22_triplet_loss import *

def test_p22_triplet_loss():

    torch.manual_seed(0)
    with step("loss is 0 when negative is far from anchor"):
        a = torch.zeros(4, 3); p = torch.zeros(4, 3); n = torch.full((4, 3), 5.0)
        assert triplet_margin_loss(a, p, n, margin=1.0).item() == 0.0
    with step("matches F.triplet_margin_loss"):
        a = torch.randn(8, 16); p = torch.randn(8, 16); n = torch.randn(8, 16)
        ref = F.triplet_margin_loss(a, p, n, margin=1.0, p=2, reduction="mean")
        got = triplet_margin_loss(a, p, n, margin=1.0)
        assert torch.allclose(got, ref, atol=1e-5)
    with step("reduction shape and sum=mean*N"):
        a = torch.randn(8, 16); p = torch.randn(8, 16); n = torch.randn(8, 16)
        none_out = triplet_margin_loss(a, p, n, margin=1.0, reduction="none")
        assert none_out.shape == (8,)
        sum_out = triplet_margin_loss(a, p, n, margin=1.0, reduction="sum")
        assert torch.isclose(sum_out, none_out.sum(), atol=1e-5)
    with step("hardest-triplet: 0 loss when classes well-separated"):
        e = torch.tensor([
            [0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
            [5.0, 5.0], [5.1, 5.0], [5.0, 5.1],
        ])
        y = torch.tensor([0, 0, 0, 1, 1, 1])
        loss = hardest_triplet_loss(e, y, margin=0.5)
        assert loss.item() == 0.0
    with step("hardest-triplet: positive loss when classes overlap"):
        e2 = torch.tensor([[0., 0.], [0.5, 0.], [0.6, 0.], [1.2, 0.]])
        y2 = torch.tensor([0, 0, 1, 1])
        loss2 = hardest_triplet_loss(e2, y2, margin=2.0)
        assert loss2.item() > 0

