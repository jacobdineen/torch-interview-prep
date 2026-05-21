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
from p47c_training import *

def test_p47c_training():
    torch.manual_seed(0)
    net = SmallCNN(num_classes=10)
    pass
    pass
    with step('training loss decreases over 20 Adam steps'):
        x = torch.randn(4, 1, 28, 28)
        y = torch.randint(0, 10, (4,))
        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        losses = []
        for _ in range(20):
            opt.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]
