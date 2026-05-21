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
from p08c_scatter_sum_rows import *

def test_p08c_scatter_sum_rows():
    pass
    pass
    with step('scatter_sum_rows sums values by row index (segment sum)'):
        values = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        index = torch.tensor([0, 1, 0, 2])
        out = scatter_sum_rows(values, index, num_rows=3)
        assert out.shape == (3, 2)
        assert torch.allclose(out, torch.tensor([[4.0, 4.0], [2.0, 2.0], [4.0, 4.0]]))
