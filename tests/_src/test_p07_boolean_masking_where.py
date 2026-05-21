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
from p07_boolean_masking_where import *

def test_p07_boolean_masking_where():

    x = torch.tensor([-1.0, 0.0, 2.0, -3.0, 4.0])
    with step("clip_negatives replaces negatives with 0"):
        assert torch.equal(clip_negatives(x), torch.tensor([0.0, 0.0, 2.0, 0.0, 4.0]))
    with step("count_above counts strictly-greater entries"):
        assert count_above(x, 0.0) == 2
    with step("replace_nan substitutes a value for NaN entries"):
        nx = torch.tensor([1.0, float("nan"), 3.0, float("nan")])
        out = replace_nan(nx, -1.0)
        assert torch.equal(out, torch.tensor([1.0, -1.0, 3.0, -1.0]))
    with step("masked_mean averages over true positions of the mask"):
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        assert torch.isclose(masked_mean(y, mask), torch.tensor(2.0))

