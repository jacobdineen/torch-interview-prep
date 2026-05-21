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
from p35e_epoch_mean import *

def test_p35e_epoch_mean():
    ds = SyntheticDataset(100, 4, seed=42)
    pass
    pass
    pass
    pass
    with step('epoch_mean equals mean of all ys'):
        loader = make_loader(ds, batch_size=8, shuffle=False)
        all_ys = torch.stack([ds[i][1] for i in range(len(ds))])
        expected = all_ys.mean()
        got = epoch_mean(loader)
        assert torch.isclose(got, expected, atol=1e-05)
