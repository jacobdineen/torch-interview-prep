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
from p35_custom_dataset_dataloader import *

def test_p35_custom_dataset_dataloader():

    ds = SyntheticDataset(100, 4, seed=42)
    with step("dataset length matches num_samples"):
        assert len(ds) == 100
    with step("__getitem__ returns (x, y) of right shape and y = sum(x^2)"):
        x, y = ds[0]
        assert x.shape == (4,) and y.shape == ()
        assert torch.isclose(y, (x ** 2).sum())
    with step("seeded dataset is deterministic"):
        ds2 = SyntheticDataset(100, 4, seed=42)
        x2, y2 = ds2[0]
        x, y = ds[0]
        assert torch.equal(x, x2) and torch.equal(y, y2)
    with step("make_loader produces correctly-shaped batches"):
        loader = make_loader(ds, batch_size=8, shuffle=False)
        xb, yb = next(iter(loader))
        assert xb.shape == (8, 4) and yb.shape == (8,)
    with step("epoch_mean equals mean of all ys"):
        loader = make_loader(ds, batch_size=8, shuffle=False)
        all_ys = torch.stack([ds[i][1] for i in range(len(ds))])
        expected = all_ys.mean()
        got = epoch_mean(loader)
        assert torch.isclose(got, expected, atol=1e-5)

