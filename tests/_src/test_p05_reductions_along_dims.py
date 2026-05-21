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
from p05_reductions_along_dims import *

def test_p05_reductions_along_dims():

    torch.manual_seed(0)
    x = torch.randn(8, 3, 16, 16)
    with step("per_channel_mean averages over batch and spatial dims"):
        pcm = per_channel_mean(x)
        assert pcm.shape == (3,)
        assert torch.allclose(pcm, x.mean(dim=(0, 2, 3)), atol=1e-6)
    with step("per_batch_max takes the max over all non-batch dims"):
        y = torch.randn(4, 5, 6)
        pbm = per_batch_max(y)
        assert pbm.shape == (4,)
        assert torch.allclose(pbm, y.flatten(1).max(dim=1).values)
    with step("normalize_along_dim subtracts mean and divides by std along dim"):
        z = torch.randn(3, 5)
        nz = normalize_along_dim(z, dim=1)
        assert torch.allclose(nz.mean(dim=1), torch.zeros(3), atol=1e-5)
        assert torch.allclose(nz.std(dim=1, unbiased=False), torch.ones(3), atol=1e-4)
    with step("rowwise_argmax returns the index of max in each row"):
        a = torch.tensor([[1., 3., 2.], [9., 0., 5.]])
        assert torch.equal(rowwise_argmax(a), torch.tensor([1, 0]))

