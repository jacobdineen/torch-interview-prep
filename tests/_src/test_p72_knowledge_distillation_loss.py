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
import torch.nn.functional as F
from p72_knowledge_distillation_loss import *

def test_p72_knowledge_distillation_loss():

    torch.manual_seed(0)
    N, C = 8, 5
    s = torch.randn(N, C, requires_grad=True)
    t = torch.randn(N, C)
    y = torch.randint(0, C, (N,))
    with step("alpha=1 equals plain CE"):
        loss = kd_loss(s, t, y, T=2.0, alpha=1.0)
        assert torch.allclose(loss, F.cross_entropy(s, y), atol=1e-6)
    with step("alpha=0 equals T^2 * KL(softmax(teacher/T) || softmax(student/T))"):
        T = 3.0
        teach_logp = F.log_softmax(t / T, dim=-1)
        teach_p = teach_logp.exp()
        student_logp = F.log_softmax(s / T, dim=-1)
        expected = (T * T) * (teach_p * (teach_logp - student_logp)).sum(dim=-1).mean()
        assert torch.allclose(kd_loss(s, t, y, T=T, alpha=0.0), expected, atol=1e-5)
    with step("when teacher == student, KL term is 0; loss = alpha * CE"):
        loss = kd_loss(s, s.detach(), y, T=3.0, alpha=0.5)
        assert torch.allclose(loss, 0.5 * F.cross_entropy(s, y), atol=1e-5)
    with step("gradients flow into student logits"):
        loss = kd_loss(s, t, y, T=3.0, alpha=0.5)
        loss.backward()
        assert s.grad is not None and torch.isfinite(s.grad).all()

