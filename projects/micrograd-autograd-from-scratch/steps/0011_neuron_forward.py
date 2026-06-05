"""
Step 0011: neuron_forward

Part 4 — Neural Network
Compute tanh of the weighted sum of inputs x plus the neuron's bias.

neuron["w"] is the list of weight Values and neuron["b"] is the bias Value; return tanh(neuron["b"] + sum(w_i * x_i)) pairing neuron["w"] with x in order (build the sum with v_mul/v_add and apply v_tanh).

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


def neuron_forward(neuron, x):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
