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
