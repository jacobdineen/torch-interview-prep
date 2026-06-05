"""Hidden reference implementations for mlp-in-jax-from-scratch.

One top-level def per step, named exactly as in spec.STEPS, each with a one-line
docstring. Steps call earlier reference functions directly -- shared namespace.
Functional JAX: explicit PRNG keys, pure parameter updates.
"""
import jax
import jax.numpy as jnp

def make_prng_key(seed):
    """Create a PRNG key from an integer seed."""
    return jax.random.PRNGKey(seed)


def split_prng_key(key, num=2):
    """Split a PRNG key into `num` independent keys."""
    return jax.random.split(key, num)


def sample_normal_matrix(key, shape):
    """Sample a standard normal array of the given shape."""
    return jax.random.normal(key, shape)

def sample_input_features(key, n_samples, n_features):
    """Draw a standard-normal feature matrix of shape (n_samples, n_features)."""
    return jax.random.normal(key, (n_samples, n_features))


def assign_class_labels(X, n_classes):
    """Label each row by argmax over sums of n_classes contiguous feature-column groups."""
    n_features = X.shape[1]
    groups = jnp.array_split(jnp.arange(n_features), n_classes)
    group_sums = jnp.stack([X[:, g].sum(axis=1) for g in groups], axis=1)
    return jnp.argmax(group_sums, axis=1).astype(jnp.int32)


def one_hot_encode_labels(labels, n_classes):
    """Return float one-hot rows of the integer labels with n_classes columns."""
    return jax.nn.one_hot(labels, n_classes)

def init_linear_layer(key, in_dim, out_dim, scale=0.1):
    """Initialize one linear layer: W = scale * normal((in_dim, out_dim)), b = zeros(out_dim)."""
    W = scale * jax.random.normal(key, (in_dim, out_dim))
    b = jnp.zeros((out_dim,))
    return (W, b)


def init_mlp_params(key, layer_sizes):
    """Initialize a list of (W, b) layers by splitting keys across consecutive layer sizes."""
    num_layers = len(layer_sizes) - 1
    keys = jax.random.split(key, num_layers)
    params = []
    for i in range(num_layers):
        params.append(init_linear_layer(keys[i], layer_sizes[i], layer_sizes[i + 1]))
    return params

def linear_forward(layer, x):
    """Apply an affine layer (W, b) to x: x @ W + b."""
    W, b = layer
    return x @ W + b


def relu_activation(x):
    """Elementwise ReLU: max(x, 0)."""
    return jnp.maximum(x, 0)


def softmax_probabilities(logits):
    """Numerically stable softmax over the last axis."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp = jnp.exp(shifted)
    return exp / jnp.sum(exp, axis=-1, keepdims=True)


def mlp_forward(params, x):
    """Run the MLP: linear+ReLU on every layer except the last, which is linear only."""
    h = x
    for layer in params[:-1]:
        h = relu_activation(linear_forward(layer, h))
    return linear_forward(params[-1], h)

def log_softmax_logits(logits):
    """Compute the numerically stable log-softmax of logits over the last axis."""
    shifted = logits - jnp.max(logits, axis=-1, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=-1, keepdims=True))


def cross_entropy_loss(logits, one_hot_targets):
    """Compute the mean cross-entropy loss between logits and one-hot targets."""
    log_probs = log_softmax_logits(logits)
    return jnp.mean(-jnp.sum(one_hot_targets * log_probs, axis=-1))


def classification_accuracy(logits, labels):
    """Compute the fraction of rows whose argmax prediction matches the integer labels."""
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)

def loss_fn_of_params(params, x, one_hot_targets):
    """Cross-entropy loss as a pure function of params, for jax.grad to differentiate."""
    return cross_entropy_loss(mlp_forward(params, x), one_hot_targets)


def compute_param_grads(params, x, one_hot_targets):
    """Gradient of the loss with respect to params (same pytree structure)."""
    return jax.grad(loss_fn_of_params)(params, x, one_hot_targets)


def sgd_update_params(params, grads, lr):
    """Apply one SGD step: subtract lr * grad from each parameter leaf."""
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

def training_step(params, x, one_hot_targets, lr):
    """Perform one SGD step: compute the loss and grads, update params, return (new_params, loss)."""
    loss = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    new_params = sgd_update_params(params, grads, lr)
    return new_params, loss


def train_mlp(params, X, one_hot_targets, lr, n_steps):
    """Run n_steps of full-batch SGD, returning the final params and the list of per-step losses."""
    loss_history = []
    for _ in range(n_steps):
        params, loss = training_step(params, X, one_hot_targets, lr)
        loss_history.append(float(loss))
    return params, loss_history


def predict_classes(params, X):
    """Return the predicted integer class for each row as the argmax of the MLP logits."""
    logits = mlp_forward(params, X)
    return jnp.argmax(logits, axis=-1)
