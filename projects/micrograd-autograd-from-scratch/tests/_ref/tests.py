"""Hidden tests for micrograd-autograd-from-scratch. One test_<id>_<name>(ns) per step; deterministic, independent oracles."""
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

def test_0001_v_add(ns):
    Value = ns["Value"]
    v_add = ns["v_add"]

    a = Value(2.5)
    b = Value(-1.25)
    out = v_add(a, b)
    assert abs(out.data - (2.5 + -1.25)) < 1e-12

    out.grad = 1.0
    out._backward()
    # d(a+b)/da = 1, d(a+b)/db = 1
    assert abs(a.grad - 1.0) < 1e-9
    assert abs(b.grad - 1.0) < 1e-9

    # numeric check
    eps = 1e-6
    f = lambda x, y: x + y
    da = (f(2.5 + eps, -1.25) - f(2.5 - eps, -1.25)) / (2 * eps)
    db = (f(2.5, -1.25 + eps) - f(2.5, -1.25 - eps)) / (2 * eps)
    assert abs(a.grad - da) < 1e-4
    assert abs(b.grad - db) < 1e-4


def test_0002_v_mul(ns):
    Value = ns["Value"]
    v_mul = ns["v_mul"]

    a = Value(3.0)
    b = Value(-2.0)
    out = v_mul(a, b)
    assert abs(out.data - (3.0 * -2.0)) < 1e-12

    out.grad = 1.0
    out._backward()
    # d(a*b)/da = b, d(a*b)/db = a
    assert abs(a.grad - (-2.0)) < 1e-9
    assert abs(b.grad - 3.0) < 1e-9

    eps = 1e-6
    f = lambda x, y: x * y
    da = (f(3.0 + eps, -2.0) - f(3.0 - eps, -2.0)) / (2 * eps)
    db = (f(3.0, -2.0 + eps) - f(3.0, -2.0 - eps)) / (2 * eps)
    assert abs(a.grad - da) < 1e-4
    assert abs(b.grad - db) < 1e-4


def test_0003_v_pow(ns):
    Value = ns["Value"]
    v_pow = ns["v_pow"]

    a = Value(1.7)
    n = 3
    out = v_pow(a, n)
    assert abs(out.data - (1.7 ** 3)) < 1e-12

    out.grad = 1.0
    out._backward()
    # d(a**n)/da = n*a**(n-1)
    assert abs(a.grad - (3 * 1.7 ** 2)) < 1e-9

    eps = 1e-6
    f = lambda x: x ** n
    da = (f(1.7 + eps) - f(1.7 - eps)) / (2 * eps)
    assert abs(a.grad - da) < 1e-4

    # fractional exponent
    a2 = Value(4.0)
    out2 = v_pow(a2, 0.5)
    assert abs(out2.data - 2.0) < 1e-12
    out2.grad = 1.0
    out2._backward()
    da2 = (4.0 ** 0.5 + 0) and (((4.0 + eps) ** 0.5 - (4.0 - eps) ** 0.5) / (2 * eps))
    assert abs(a2.grad - da2) < 1e-4


def test_0004_v_sub(ns):
    Value = ns["Value"]
    v_sub = ns["v_sub"]

    a = Value(5.5)
    b = Value(2.0)
    out = v_sub(a, b)
    assert abs(out.data - (5.5 - 2.0)) < 1e-12

    out.grad = 1.0
    out._backward()
    # d(a-b)/da = 1, d(a-b)/db = -1
    assert abs(a.grad - 1.0) < 1e-9
    assert abs(b.grad - (-1.0)) < 1e-9

    eps = 1e-6
    f = lambda x, y: x - y
    da = (f(5.5 + eps, 2.0) - f(5.5 - eps, 2.0)) / (2 * eps)
    db = (f(5.5, 2.0 + eps) - f(5.5, 2.0 - eps)) / (2 * eps)
    assert abs(a.grad - da) < 1e-4
    assert abs(b.grad - db) < 1e-4

import math


def _num_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_0005_v_relu(ns):
    Value = ns["Value"]
    v_relu = ns["v_relu"]

    # positive branch
    a = Value(1.7)
    out = v_relu(a)
    assert abs(out.data - max(0.0, 1.7)) < 1e-12
    out.grad = 1.0
    out._backward()
    g = _num_grad(lambda x: max(0.0, x), 1.7)
    assert abs(a.grad - g) < 1e-4

    # negative branch
    b = Value(-2.3)
    out2 = v_relu(b)
    assert abs(out2.data - 0.0) < 1e-12
    out2.grad = 1.0
    out2._backward()
    assert abs(b.grad - 0.0) < 1e-4

    # gradient accumulation (+=)
    c = Value(0.5)
    oc = v_relu(c)
    c.grad = 3.0
    oc.grad = 1.0
    oc._backward()
    assert abs(c.grad - (3.0 + 1.0)) < 1e-9


def test_0006_v_tanh(ns):
    Value = ns["Value"]
    v_tanh = ns["v_tanh"]

    a = Value(0.8)
    out = v_tanh(a)
    assert abs(out.data - math.tanh(0.8)) < 1e-12
    out.grad = 1.0
    out._backward()
    g = _num_grad(math.tanh, 0.8)
    assert abs(a.grad - g) < 1e-4

    # negative input
    b = Value(-1.5)
    ob = v_tanh(b)
    assert abs(ob.data - math.tanh(-1.5)) < 1e-12
    ob.grad = 1.0
    ob._backward()
    gb = _num_grad(math.tanh, -1.5)
    assert abs(b.grad - gb) < 1e-4

    # accumulation
    c = Value(0.2)
    oc = v_tanh(c)
    c.grad = -2.0
    oc.grad = 1.0
    oc._backward()
    expected = -2.0 + (1.0 - math.tanh(0.2) ** 2)
    assert abs(c.grad - expected) < 1e-9


def test_0007_v_exp(ns):
    Value = ns["Value"]
    v_exp = ns["v_exp"]

    a = Value(1.1)
    out = v_exp(a)
    assert abs(out.data - math.exp(1.1)) < 1e-12
    out.grad = 1.0
    out._backward()
    g = _num_grad(math.exp, 1.1)
    assert abs(a.grad - g) < 1e-3

    # negative input
    b = Value(-0.7)
    ob = v_exp(b)
    assert abs(ob.data - math.exp(-0.7)) < 1e-12
    ob.grad = 1.0
    ob._backward()
    gb = _num_grad(math.exp, -0.7)
    assert abs(b.grad - gb) < 1e-4

    # accumulation with upstream grad
    c = Value(0.3)
    oc = v_exp(c)
    c.grad = 5.0
    oc.grad = 2.0
    oc._backward()
    expected = 5.0 + math.exp(0.3) * 2.0
    assert abs(c.grad - expected) < 1e-9

import math


def _fd_grad(f, x, eps=1e-6):
    """Central finite-difference derivative of f at scalar x."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_0008_build_topo(ns):
    Value = ns["Value"]
    build_topo = ns["build_topo"]

    # Build a small graph by hand: out = (a*b) + a  (a used twice)
    a = Value(2.0)
    b = Value(3.0)

    prod = Value(a.data * b.data, (a, b), "*")
    out = Value(prod.data + a.data, (prod, a), "+")

    topo = build_topo(out)

    # Each node appears exactly once
    assert len(topo) == len(set(id(n) for n in topo))
    # All graph nodes present
    assert set(id(n) for n in topo) == {id(a), id(b), id(prod), id(out)}
    # Root is last
    assert topo[-1] is out

    # Every child appears before any parent that depends on it
    pos = {id(n): i for i, n in enumerate(topo)}
    for n in topo:
        for child in n._prev:
            assert pos[id(child)] < pos[id(n)]


def test_0009_backward(ns):
    Value = ns["Value"]
    backward = ns["backward"]

    # Graph with a node used twice: out = a*a + a  -> d/da = 2a + 1
    def build(av):
        a = Value(av)
        prod = Value(a.data * a.data, (a, a), "*")

        def _bw_prod():
            a.grad += a.data * prod.grad
            a.grad += a.data * prod.grad
        prod._backward = _bw_prod

        out = Value(prod.data + a.data, (prod, a), "+")

        def _bw_out():
            prod.grad += out.grad
            a.grad += out.grad
        out._backward = _bw_out
        return a, out

    a, out = build(2.0)
    backward(out)

    # Root grad seeded
    assert out.grad == 1.0

    # Analytic: d(a*a + a)/da = 2a + 1 = 5
    assert abs(a.grad - 5.0) < 1e-9

    # Finite-difference check on forward value f(a) = a*a + a
    def f(av):
        return av * av + av
    num = _fd_grad(f, 2.0)
    assert abs(a.grad - num) < 1e-4

    # Determinism: rebuilding and re-running gives the same grad
    a2, out2 = build(2.0)
    backward(out2)
    assert abs(a2.grad - a.grad) < 1e-12

import math
import random as _random


def test_0010_init_neuron(ns):
    Value = ns["Value"]
    init_neuron = ns["init_neuron"]

    nin, seed = 4, 7
    neuron = init_neuron(nin, seed=seed)

    # structure
    assert set(neuron.keys()) == {"w", "b"}
    assert isinstance(neuron["w"], list)
    assert len(neuron["w"]) == nin
    for wi in neuron["w"]:
        assert isinstance(wi, Value)
    assert isinstance(neuron["b"], Value)

    # bias is exactly zero
    assert neuron["b"].data == 0.0

    # weights match an independent random.Random(seed) draw in [-1,1]
    rng = _random.Random(seed)
    expected = [rng.uniform(-1, 1) for _ in range(nin)]
    for wi, e in zip(neuron["w"], expected):
        assert abs(wi.data - e) < 1e-12
        assert -1.0 <= wi.data <= 1.0

    # determinism: same seed -> same weights
    neuron2 = init_neuron(nin, seed=seed)
    for a, b in zip(neuron["w"], neuron2["w"]):
        assert a.data == b.data

    # different seed -> (very likely) different weights
    other = init_neuron(nin, seed=seed + 1)
    assert any(abs(a.data - b.data) > 1e-12
               for a, b in zip(neuron["w"], other["w"]))


def test_0011_neuron_forward(ns):
    Value = ns["Value"]
    neuron_forward = ns["neuron_forward"]
    backward = ns["backward"]

    # fixed, small neuron and input
    ws = [0.5, -1.5, 2.0]
    bias = 0.3
    neuron = {"w": [Value(w) for w in ws], "b": Value(bias)}
    xs = [1.0, -2.0, 0.5]
    x = [Value(v) for v in xs]

    out = neuron_forward(neuron, x)
    assert isinstance(out, Value)

    # forward: tanh(sum w_i x_i + b)
    pre = sum(w * xi for w, xi in zip(ws, xs)) + bias
    assert abs(out.data - math.tanh(pre)) < 1e-9

    # backward: check grad w.r.t. each weight and the bias via finite diff
    backward(out)

    def f(weights, b):
        p = sum(w * xi for w, xi in zip(weights, xs)) + b
        return math.tanh(p)

    eps = 1e-6
    for i in range(len(ws)):
        wp = list(ws); wp[i] += eps
        wm = list(ws); wm[i] -= eps
        num = (f(wp, bias) - f(wm, bias)) / (2 * eps)
        assert abs(neuron["w"][i].grad - num) < 1e-4

    numb = (f(ws, bias + eps) - f(ws, bias - eps)) / (2 * eps)
    assert abs(neuron["b"].grad - numb) < 1e-4

    # grad w.r.t. inputs too
    for i in range(len(xs)):
        xp = list(xs); xp[i] += eps
        xm = list(xs); xm[i] -= eps
        pp = sum(w * v for w, v in zip(ws, xp)) + bias
        pm = sum(w * v for w, v in zip(ws, xm)) + bias
        num = (math.tanh(pp) - math.tanh(pm)) / (2 * eps)
        assert abs(x[i].grad - num) < 1e-4


def test_0012_layer_forward(ns):
    Value = ns["Value"]
    layer_forward = ns["layer_forward"]

    # two neurons, 2 inputs
    n0 = {"w": [Value(1.0), Value(0.0)], "b": Value(0.0)}
    n1 = {"w": [Value(0.0), Value(2.0)], "b": Value(0.5)}
    layer = [n0, n1]
    xs = [0.3, -0.4]
    x = [Value(v) for v in xs]

    outs = layer_forward(layer, x)
    assert isinstance(outs, list)
    assert len(outs) == len(layer)
    for o in outs:
        assert isinstance(o, Value)

    # each output equals tanh of its own neuron's preactivation
    exp0 = math.tanh(1.0 * xs[0] + 0.0 * xs[1] + 0.0)
    exp1 = math.tanh(0.0 * xs[0] + 2.0 * xs[1] + 0.5)
    assert abs(outs[0].data - exp0) < 1e-9
    assert abs(outs[1].data - exp1) < 1e-9


def test_0013_mlp_forward(ns):
    Value = ns["Value"]
    mlp_forward = ns["mlp_forward"]

    # layer 0: 2 inputs -> 2 neurons ; layer 1: 2 inputs -> 1 neuron
    l0 = [
        {"w": [Value(0.5), Value(-0.5)], "b": Value(0.1)},
        {"w": [Value(-1.0), Value(1.0)], "b": Value(-0.2)},
    ]
    l1 = [
        {"w": [Value(2.0), Value(0.5)], "b": Value(0.0)},
    ]
    mlp = [l0, l1]
    xs = [0.7, -0.3]
    x = [Value(v) for v in xs]

    out = mlp_forward(mlp, x)
    assert isinstance(out, list)
    assert len(out) == 1  # final layer width
    assert isinstance(out[0], Value)

    # recompute the whole network by hand
    def neuron(w, b, inp):
        return math.tanh(sum(wi * xi for wi, xi in zip(w, inp)) + b)

    h0 = neuron([0.5, -0.5], 0.1, xs)
    h1 = neuron([-1.0, 1.0], -0.2, xs)
    y = neuron([2.0, 0.5], 0.0, [h0, h1])

    assert abs(out[0].data - y) < 1e-9

    # output width equals last layer's neuron count for a different shape
    l0b = [{"w": [Value(0.1)], "b": Value(0.0)}]
    l1b = [{"w": [Value(0.2)], "b": Value(0.0)},
           {"w": [Value(0.3)], "b": Value(0.0)},
           {"w": [Value(0.4)], "b": Value(0.0)}]
    out2 = mlp_forward([l0b, l1b], [Value(1.0)])
    assert len(out2) == 3

def test_0014_parameters(ns):
    Value = ns["Value"]
    parameters = ns["parameters"]
    # Build a fake mlp: 2 layers, neurons with explicit weights/bias.
    n1 = {"w": [Value(1.0), Value(2.0)], "b": Value(0.5)}
    n2 = {"w": [Value(3.0), Value(4.0)], "b": Value(0.6)}
    n3 = {"w": [Value(5.0)], "b": Value(0.7)}
    mlp = [[n1, n2], [n3]]
    params = parameters(mlp)
    # Expect order: n1.w[0], n1.w[1], n1.b, n2.w[0], n2.w[1], n2.b, n3.w[0], n3.b
    expected_data = [1.0, 2.0, 0.5, 3.0, 4.0, 0.6, 5.0, 0.7]
    assert len(params) == len(expected_data)
    for p, d in zip(params, expected_data):
        assert isinstance(p, Value)
        assert p.data == d
    # Identity: returned objects are the very same Value instances (in order).
    assert params[0] is n1["w"][0]
    assert params[2] is n1["b"]
    assert params[-1] is n3["b"]


def test_0015_mse_loss(ns):
    Value = ns["Value"]
    mse_loss = ns["mse_loss"]
    backward = ns["backward"]
    preds_raw = [0.5, -1.0, 2.0]
    targets = [1.0, 0.0, 1.5]
    preds = [Value(p) for p in preds_raw]
    loss = mse_loss(preds, targets)
    # Independent oracle: sum of squared errors.
    expected = sum((p - t) ** 2 for p, t in zip(preds_raw, targets))
    assert abs(loss.data - expected) < 1e-9
    # Gradient check: dL/dpred_i = 2*(pred_i - target_i), via finite differences.
    backward(loss)
    eps = 1e-6
    for i in range(len(preds_raw)):
        def f(val):
            pr = list(preds_raw)
            pr[i] = val
            return sum((p - t) ** 2 for p, t in zip(pr, targets))
        num = (f(preds_raw[i] + eps) - f(preds_raw[i] - eps)) / (2 * eps)
        assert abs(preds[i].grad - num) < 1e-4


def test_0016_zero_grad(ns):
    Value = ns["Value"]
    zero_grad = ns["zero_grad"]
    params = [Value(1.0), Value(2.0), Value(3.0)]
    for p in params:
        p.grad = 7.5
    ret = zero_grad(params)
    assert ret is None
    for p in params:
        assert p.grad == 0.0
        # Data must be untouched.
    assert [p.data for p in params] == [1.0, 2.0, 3.0]


def test_0017_sgd_step(ns):
    Value = ns["Value"]
    sgd_step = ns["sgd_step"]
    params = [Value(1.0), Value(2.0), Value(-3.0)]
    grads = [0.5, -1.0, 2.0]
    for p, g in zip(params, grads):
        p.grad = g
    lr = 0.1
    expected = [d - lr * g for d, g in zip([1.0, 2.0, -3.0], grads)]
    ret = sgd_step(params, lr)
    assert ret is None
    for p, e in zip(params, expected):
        assert abs(p.data - e) < 1e-12
    # Grads themselves are not modified by sgd_step.
    for p, g in zip(params, grads):
        assert p.grad == g


def test_0018_train(ns):
    Value = ns["Value"]
    init_neuron = ns["init_neuron"]
    train = ns["train"]
    # Build a tiny single-output MLP: 2 inputs -> hidden(3) -> output(1).
    layer1 = [init_neuron(2, seed=i) for i in range(3)]
    layer2 = [init_neuron(3, seed=10)]
    mlp = [layer1, layer2]
    # Tiny dataset.
    X = [[0.5, -0.5], [1.0, 1.0], [-1.0, 0.5], [0.2, -0.8]]
    Y = [0.5, -0.5, 0.5, -0.5]
    losses = train(mlp, X, Y, lr=0.05, n_steps=40)
    assert isinstance(losses, list)
    assert len(losses) == 40
    for l in losses:
        assert isinstance(l, float)
        assert l == l  # not NaN
        assert l != float("inf") and l != float("-inf")
        assert l >= 0.0
    # Loss must decrease substantially.
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.5 * losses[0]
    # Determinism: same seeds + same data -> identical loss trajectory.
    layer1b = [init_neuron(2, seed=i) for i in range(3)]
    layer2b = [init_neuron(3, seed=10)]
    mlpb = [layer1b, layer2b]
    lossesb = train(mlpb, X, Y, lr=0.05, n_steps=40)
    # Same seeds + data -> same trajectory. Allow a tiny tolerance: build_topo walks
    # the set-valued _prev, so sibling gradients accumulate in an id-dependent float
    # order that can differ in the last ULP between two independently built graphs.
    assert len(lossesb) == len(losses)
    for a, b in zip(losses, lossesb):
        assert abs(a - b) < 1e-9
