"""Build spec for the micrograd-autograd-from-scratch project."""

TITLE = 'Build Your Own Autograd Engine (micrograd)'

PARTS = [
    ("Scalar Arithmetic", "Implement add, multiply, power, and subtract on Value nodes, each wiring up its local backward closure."),
    ("Activations", "Implement ReLU, tanh, and exp on Value nodes with their local gradients."),
    ("Reverse-Mode Autodiff", "Topologically sort the graph and run reverse-mode backprop from a scalar output."),
    ("Neural Network", "Build neurons, layers, and an MLP out of Value nodes."),
    ("Training", "Collect parameters, an MSE loss, zero_grad, an SGD step, and the training loop."),
]

STEPS = [
    ("v_add", 0), ("v_mul", 0), ("v_pow", 0), ("v_sub", 0),
    ("v_relu", 1), ("v_tanh", 1), ("v_exp", 1),
    ("build_topo", 2), ("backward", 2),
    ("init_neuron", 3), ("neuron_forward", 3), ("layer_forward", 3), ("mlp_forward", 3),
    ("parameters", 4), ("mse_loss", 4), ("zero_grad", 4), ("sgd_step", 4), ("train", 4),
]


def step_id(i):
    return f"{i:04d}"


PRIMER = '''
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
`uv run python projects.py micrograd-autograd-from-scratch` (or the outline drawer) to see all signatures.'''
