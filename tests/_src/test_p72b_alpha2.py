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
import torch.nn.functional as F
from p72b_alpha2 import *

def test_p72b_alpha2():
    torch.manual_seed(0)
    N, C = (8, 5)
    s = torch.randn(N, C, requires_grad=True)
    t = torch.randn(N, C)
    y = torch.randint(0, C, (N,))
    pass
    with step('alpha=0 equals T^2 * KL(softmax(teacher/T) || softmax(student/T))'):
        T = 3.0
        teach_logp = F.log_softmax(t / T, dim=-1)
        teach_p = teach_logp.exp()
        student_logp = F.log_softmax(s / T, dim=-1)
        expected = T * T * (teach_p * (teach_logp - student_logp)).sum(dim=-1).mean()
        assert torch.allclose(kd_loss(s, t, y, T=T, alpha=0.0), expected, atol=1e-05)
    pass
    pass
