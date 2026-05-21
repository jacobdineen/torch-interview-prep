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
import torch.nn as nn
from p37_train_mlp_regression import *

def test_p37_train_mlp_regression():

    torch.manual_seed(0)

    # Synthetic data: y = X @ W_true + small_noise + sin(x_0)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, 4, generator=g)
    W_true = torch.randn(4, 1, generator=g)
    y = x @ W_true + 0.1 * torch.randn(512, 1, generator=g) + torch.sin(x[:, :1])

    model = build_mlp(in_dim=4, hidden_dims=[32, 32], out_dim=1)
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    relus = [m for m in model.modules() if isinstance(m, nn.ReLU)]
    with step("MLP has 3 Linears + 2 ReLUs in between"):
        assert len(linears) == 3
        assert len(relus) == 2
    with step("fit reduces loss by at least 20% over 10 epochs"):
        initial_loss = ((model(x) - y) ** 2).mean().item()
        final_loss = fit(model, x, y, epochs=10, lr=1e-2, batch_size=64)
        assert final_loss < initial_loss * 0.8, f"final={final_loss}, initial={initial_loss}"

