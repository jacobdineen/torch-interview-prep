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
from p02_indexing_slicing import *

def test_p02_indexing_slicing():

    x = torch.arange(20).reshape(4, 5)
    with step("get_row(x, 2) returns the 3rd row"):
        assert torch.equal(get_row(x, 2), torch.tensor([10, 11, 12, 13, 14]))
    sq = torch.arange(9).reshape(3, 3)
    with step("get_diagonal returns the matrix diagonal"):
        assert torch.equal(get_diagonal(sq), torch.tensor([0, 4, 8]))
    with step("every_other_col returns columns 0, 2, 4, ..."):
        assert torch.equal(every_other_col(x), x[:, ::2])
    with step("select_rows picks rows by index"):
        idx = torch.tensor([3, 0])
        assert torch.equal(select_rows(x, idx), x[[3, 0]])
    with step("top_left_block(x, 2) returns the top-left 2x2"):
        assert torch.equal(top_left_block(x, 2), x[:2, :2])

