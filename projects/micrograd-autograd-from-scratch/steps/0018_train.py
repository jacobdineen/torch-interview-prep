"""
Step 0018: train

Part 5 — Training
Run n_steps of full-batch SGD on the MLP and return the list of per-step loss floats.

X is a list of input vectors (each a list of floats); Y a list of target floats (single-output MLP). Each step: zero_grad(parameters(mlp)); for every input, WRAP its floats as [Value(xi) for xi in x] (mlp_forward operates on Value nodes, not raw floats) and take mlp_forward(mlp, xv)[0] as the scalar prediction; mse_loss over all predictions vs Y; backward; sgd_step. Return the list of per-step loss floats (loss.data), which should decrease.

Provided for you (already defined in your namespace at grade time — use it, do NOT redefine):

  class Value: a scalar node in the autograd graph. Construct with
      Value(data, _children=(), _op="").
    Attributes:
      .data       float — the scalar value
      .grad       float — the accumulated dL/d(self); starts at 0.0
      ._prev      set of the parent Value nodes this one was built from
      ._op        short str label (e.g. "+", "*", "tanh"); for debugging
      ._backward  a zero-arg closure. Each op you write sets it to ADD (+=) this
                  node's local gradient into each input's .grad, scaled by out.grad.

Pattern for an op: out = Value(result, (inputs...), "op"); define a local _backward()
that pushes gradient to the inputs; set out._backward = _backward; return out.
backward(root) seeds root.grad = 1.0 and runs every _backward in reverse topological order.
Every other function in this project is also available in your namespace at grade
time — call earlier steps by name; you do not import them. Run
`uv run python projects.py micrograd-autograd-from-scratch` (or the outline drawer) to see all signatures."""
import math  # noqa: F401


def train(mlp, X, Y, lr, n_steps):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
