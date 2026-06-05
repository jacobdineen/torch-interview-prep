"""Hidden reference implementations for micrograd-autograd-from-scratch. Steps call earlier references
(shared namespace). One def per step with a one-line docstring."""
import math
import random  # noqa: F401


class Value:
    """A scalar node in the autograd graph: its value, its gradient, and how it was built."""

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

def v_add(a, b):
    """Return a Value holding the sum of a and b, propagating gradient to both addends."""
    out = Value(a.data + b.data, (a, b), "+")

    def _backward():
        a.grad += out.grad
        b.grad += out.grad

    out._backward = _backward
    return out


def v_mul(a, b):
    """Return a Value holding the product of a and b, with the product-rule gradient."""
    out = Value(a.data * b.data, (a, b), "*")

    def _backward():
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out._backward = _backward
    return out


def v_pow(a, n):
    """Return a Value holding a raised to the constant power n, with the power-rule gradient."""
    out = Value(a.data ** n, (a,), "**" + str(n))

    def _backward():
        a.grad += n * a.data ** (n - 1) * out.grad

    out._backward = _backward
    return out


def v_sub(a, b):
    """Return a Value holding the difference a minus b, propagating gradient with opposite signs."""
    out = Value(a.data - b.data, (a, b), "-")

    def _backward():
        a.grad += out.grad
        b.grad -= out.grad

    out._backward = _backward
    return out

def v_relu(a):
    """Apply the rectified linear unit (clamp negatives to zero) to a Value."""
    out = Value(max(0.0, a.data), (a,), "relu")

    def _backward():
        a.grad += (1.0 if out.data > 0 else 0.0) * out.grad

    out._backward = _backward
    return out


def v_tanh(a):
    """Apply the hyperbolic tangent activation to a Value."""
    t = math.tanh(a.data)
    out = Value(t, (a,), "tanh")

    def _backward():
        a.grad += (1.0 - t * t) * out.grad

    out._backward = _backward
    return out


def v_exp(a):
    """Apply the natural exponential to a Value."""
    e = math.exp(a.data)
    out = Value(e, (a,), "exp")

    def _backward():
        a.grad += e * out.grad

    out._backward = _backward
    return out

def build_topo(root):
    """Return the nodes of root's graph in depth-first postorder (children before parents, root last)."""
    topo = []
    visited = set()

    def visit(node):
        if node not in visited:
            visited.add(node)
            for child in node._prev:
                visit(child)
            topo.append(node)

    visit(root)
    return topo


def backward(root):
    """Seed root's grad to 1.0 and run each node's _backward in reverse topological order."""
    topo = build_topo(root)
    root.grad = 1.0
    for node in reversed(topo):
        node._backward()

def init_neuron(nin, seed=0):
    """Create a neuron as a dict of nin random weights in [-1,1] and a zero bias."""
    import random
    rng = random.Random(seed)
    w = [Value(rng.uniform(-1, 1)) for _ in range(nin)]
    b = Value(0.0)
    return {"w": w, "b": b}


def neuron_forward(neuron, x):
    """Compute tanh of the weighted sum of inputs x plus the neuron's bias."""
    act = neuron["b"]
    for wi, xi in zip(neuron["w"], x):
        act = v_add(act, v_mul(wi, xi))
    return v_tanh(act)


def layer_forward(layer, x):
    """Apply every neuron in the layer to the same input list x, returning their outputs."""
    return [neuron_forward(n, x) for n in layer]


def mlp_forward(mlp, x):
    """Thread the input list x through each layer in turn and return the last layer's outputs."""
    out = x
    for layer in mlp:
        out = layer_forward(layer, out)
    return out

def parameters(mlp):
    """Return a flat list of all weight and bias Values across every neuron in every layer."""
    params = []
    for layer in mlp:
        for neuron in layer:
            for w in neuron["w"]:
                params.append(w)
            params.append(neuron["b"])
    return params


def mse_loss(preds, targets):
    """Return the summed squared-error Value between predictions and target floats."""
    loss = None
    for pred, target in zip(preds, targets):
        diff = v_sub(pred, Value(target))
        sq = v_pow(diff, 2)
        loss = sq if loss is None else v_add(loss, sq)
    return loss


def zero_grad(params):
    """Reset the .grad of every parameter Value to 0.0."""
    for p in params:
        p.grad = 0.0


def sgd_step(params, lr):
    """Update each parameter's data in place by a gradient-descent step of size lr."""
    for p in params:
        p.data -= lr * p.grad


def train(mlp, X, Y, lr, n_steps):
    """Run n_steps of full-batch SGD on the MLP and return the list of per-step loss floats."""
    losses = []
    for _ in range(n_steps):
        params = parameters(mlp)
        zero_grad(params)
        preds = []
        for x in X:
            xv = [Value(xi) for xi in x]
            out = mlp_forward(mlp, xv)
            preds.append(out[0])
        loss = mse_loss(preds, Y)
        backward(loss)
        sgd_step(params, lr)
        losses.append(loss.data)
    return losses
