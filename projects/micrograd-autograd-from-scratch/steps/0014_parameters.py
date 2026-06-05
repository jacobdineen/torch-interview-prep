"""
Step 0014: parameters

Part 5 — Training
Return a flat list of all weight and bias Values across every neuron in every layer.

mlp is a list of layers, each a list of neuron dicts ("w": list of weight Values, "b": bias Value). Return the same Value objects (not copies), ordered per layer, then per neuron, emitting all of neuron["w"] in order followed by neuron["b"].

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


def parameters(mlp):
    raise NotImplementedError


if __name__ == "__main__":
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(os.path.dirname(_here)))
    sys.path.insert(0, _repo)
    from project_runner import run_step

    raise SystemExit(run_step(os.path.abspath(__file__)))
