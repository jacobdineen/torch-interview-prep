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
from p47_simple_cnn_classifier import *

def test_p47_simple_cnn_classifier():

    torch.manual_seed(0)
    net = SmallCNN(num_classes=10)
    with step("forward on (4, 1, 28, 28) produces (4, 10) logits"):
        x = torch.randn(4, 1, 28, 28)
        out = net(x)
        assert out.shape == (4, 10)
    with step("parameter count > 100k (head alone is 100k+)"):
        n = count_parameters(net)
        assert n > 100_000
    with step("training loss decreases over 20 Adam steps"):
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward(); opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]

