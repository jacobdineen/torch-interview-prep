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
from p06_concat_stack_split import *

def test_p06_concat_stack_split():

    a = torch.arange(6).reshape(3, 2)
    b = torch.arange(9).reshape(3, 3) + 100
    with step("cat_features concatenates along the feature dim"):
        c = cat_features(a, b)
        assert c.shape == (3, 5)
        assert torch.equal(c[:, :2], a) and torch.equal(c[:, 2:], b)
    with step("stack_batch creates a new leading dim"):
        items = [torch.ones(4) * i for i in range(5)]
        s = stack_batch(items)
        assert s.shape == (5, 4)
        assert torch.equal(s[2], torch.full((4,), 2.0))
    with step("split_evenly partitions into k equal chunks"):
        x = torch.arange(12)
        parts = split_evenly(x, 4)
        assert len(parts) == 4
        assert all(p.shape == (3,) for p in parts)
        assert torch.equal(torch.cat(list(parts)), x)
    with step("interleave alternates rows from a and b"):
        a2 = torch.tensor([[1, 1], [3, 3]])
        b2 = torch.tensor([[2, 2], [4, 4]])
        result = interleave(a2, b2)
        assert torch.equal(result, torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4]]))

