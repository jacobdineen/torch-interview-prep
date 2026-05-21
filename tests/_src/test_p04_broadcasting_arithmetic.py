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
from p04_broadcasting_arithmetic import *

def test_p04_broadcasting_arithmetic():

    torch.manual_seed(0)
    X = torch.randn(5, 4)
    with step("center_rows subtracts per-row mean"):
        c = center_rows(X)
        assert torch.allclose(c.mean(dim=1), torch.zeros(5), atol=1e-6)
    with step("normalize_per_sample yields per-row mean 0 std 1"):
        n = normalize_per_sample(X)
        assert torch.allclose(n.mean(dim=1), torch.zeros(5), atol=1e-5)
        assert torch.allclose(n.std(dim=1, unbiased=False), torch.ones(5), atol=1e-4)
    with step("pairwise_squared_distances computes ||a-b||^2 over D"):
        A = torch.randn(3, 4); B = torch.randn(2, 4)
        D = pairwise_squared_distances(A, B)
        assert D.shape == (3, 2)
        expected = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        assert torch.allclose(D, expected, atol=1e-5)
    with step("outer_product of 1D vectors"):
        u = torch.tensor([1.0, 2.0, 3.0]); v = torch.tensor([4.0, 5.0])
        op = outer_product(u, v)
        assert op.shape == (3, 2)
        assert torch.equal(op, torch.tensor([[4., 5.], [8., 10.], [12., 15.]]))

