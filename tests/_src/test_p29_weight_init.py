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

import math
import torch
from p29_weight_init import *

def test_p29_weight_init():

    torch.manual_seed(0)
    with step("xavier_normal_ std ~ sqrt(2/(fan_in+fan_out))"):
        w = torch.empty(64, 128)
        xavier_normal_(w, gain=1.0)
        expected = math.sqrt(2.0 / (128 + 64))
        assert abs(w.std().item() - expected) < 0.02
    with step("kaiming_normal_ fan_in,relu: std ~ sqrt(2)/sqrt(fan_in)"):
        w2 = torch.empty(64, 128)
        kaiming_normal_(w2, mode="fan_in", nonlinearity="relu")
        expected = math.sqrt(2.0) / math.sqrt(128)
        assert abs(w2.std().item() - expected) < 0.02
    with step("kaiming_normal_ fan_out,linear: std ~ 1/sqrt(fan_out)"):
        w3 = torch.empty(64, 128)
        kaiming_normal_(w3, mode="fan_out", nonlinearity="linear")
        expected = 1.0 / math.sqrt(64)
        assert abs(w3.std().item() - expected) < 0.02
    with step("kaiming_normal_ returns the tensor"):
        out = kaiming_normal_(torch.empty(8, 8))
        assert isinstance(out, torch.Tensor)

