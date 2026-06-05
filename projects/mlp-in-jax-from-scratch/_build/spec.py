"""Build spec for the mlp-in-jax-from-scratch project."""

TITLE = 'Build an MLP in JAX from Scratch'

PARTS = [
    ("PRNG & Random Sampling",
     "Set up JAX's functional PRNG: wrap seeds, split keys, and sample random tensors."),
    ("Synthetic Data & Labels",
     "Generate input features, assign deterministic class labels, and one-hot encode targets."),
    ("Parameter Initialization",
     "Initialize a single linear layer and extend to a full MLP parameter list."),
    ("Forward Pass Building Blocks",
     "Implement the linear layer, ReLU, softmax, and the full MLP forward pass."),
    ("Loss & Metrics",
     "Compute log-softmax, mean cross-entropy loss, and classification accuracy."),
    ("Autodiff & SGD Update",
     "Wrap the loss for autodiff, take gradients with jax.grad, and apply SGD updates."),
    ("Training Loop & Inference",
     "Combine the pieces into a training step, an epoch loop, and a prediction function."),
]

STEPS = [
    ("make_prng_key", 0), ("split_prng_key", 0), ("sample_normal_matrix", 0),
    ("sample_input_features", 1), ("assign_class_labels", 1), ("one_hot_encode_labels", 1),
    ("init_linear_layer", 2), ("init_mlp_params", 2),
    ("linear_forward", 3), ("relu_activation", 3), ("softmax_probabilities", 3), ("mlp_forward", 3),
    ("log_softmax_logits", 4), ("cross_entropy_loss", 4), ("classification_accuracy", 4),
    ("loss_fn_of_params", 5), ("compute_param_grads", 5), ("sgd_update_params", 5),
    ("training_step", 6), ("train_mlp", 6), ("predict_classes", 6),
]


def step_id(i):
    return f"{i:04d}"
