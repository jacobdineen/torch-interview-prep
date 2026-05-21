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
from p01_tensor_basics import *

def test_p01_tensor_basics():

    with step("make_zeros returns the requested shape and dtype"):
        z = make_zeros((2, 3), dtype=torch.float64)
        assert tuple(z.shape) == (2, 3)
        assert z.dtype == torch.float64
        assert torch.equal(z, torch.zeros(2, 3, dtype=torch.float64))
    with step("make_range returns a half-open arithmetic range"):
        r = make_range(0, 10, 2)
        assert torch.equal(r, torch.tensor([0, 2, 4, 6, 8]))
    with step("to_dtype_device converts dtype and device"):
        x = torch.zeros(2, 2, dtype=torch.float32)
        y = to_dtype_device(x, torch.int64, torch.device("cpu"))
        assert y.dtype == torch.int64
        assert y.device.type == "cpu"
    with step("tensor_info returns shape, dtype, device, numel"):
        info = tensor_info(torch.randn(3, 4))
        assert tuple(info["shape"]) == (3, 4)
        assert info["dtype"] == torch.float32
        assert info["numel"] == 12

