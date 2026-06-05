"""Build spec for the cnn-from-scratch-numpy project."""

TITLE = 'Build a Trainable CNN from Scratch in NumPy'

PARTS = [
    ("Softmax, Loss, and Metrics Primitives",
     "Build the numerically stable softmax, cross-entropy loss, and accuracy helpers used throughout the network."),
    ("Initialization and Convolution Plumbing",
     "He initialization, zero biases, padding, output-shape math, and the im2col / col2im transforms."),
    ("Layer Forward and Backward Passes",
     "Forward and backward routines for convolution, max pooling, ReLU, flatten, and linear layers."),
    ("Fused Loss and Optimizers",
     "Fuse softmax with cross-entropy for stable training and implement SGD and Adam updates."),
    ("Assembling LeNet",
     "Compose the layer primitives into conv/classifier blocks, wire the full LeNet forward/backward, and a predict helper."),
    ("Synthetic Data Pipeline",
     "Generate a synthetic image dataset and build shuffling, train/test splitting, and minibatch iteration."),
    ("Training Loop and Evaluation",
     "A training step, epoch loop, full training driver, and a held-out evaluation routine."),
]

STEPS = [
    # Part 1
    ("argmax_rows",0),("row_max",0),("row_sum",0),("exp_shifted",0),("stable_softmax",0),
    ("one_hot",0),("gather_true_class_probs",0),("cross_entropy_loss",0),("accuracy",0),
    # Part 2
    ("he_std",1),("he_init",1),("init_zero_bias",1),("pad_2d",1),("output_spatial_size",1),
    ("im2col",1),("col2im",1),
    # Part 3
    ("conv2d_forward",2),("conv2d_grad_input",2),("conv2d_grad_weights",2),("conv2d_grad_bias",2),
    ("conv2d_backward",2),("maxpool2d_forward",2),("scatter_grad_window",2),("maxpool2d_backward",2),
    ("relu_forward",2),("relu_backward",2),("flatten_forward",2),("flatten_backward",2),
    ("linear_forward",2),("linear_grad_input",2),("linear_grad_weights",2),("linear_grad_bias",2),
    ("linear_backward",2),
    # Part 4
    ("softmax_cross_entropy_forward",3),("softmax_cross_entropy_backward",3),("sgd_step",3),
    ("adam_update_m",3),("adam_update_v",3),("adam_bias_correct",3),("adam_param_step",3),("adam_step",3),
    # Part 5
    ("init_conv_layer",4),("init_linear_layer",4),("init_lenet",4),("forward_conv_block",4),
    ("forward_classifier_block",4),("lenet_forward",4),("backward_conv_block",4),
    ("backward_classifier_block",4),("lenet_backward",4),("lenet_predict",4),
    # Part 6
    ("build_synthetic_image_dataset",5),("shuffle_indices",5),("train_test_split",5),("iterate_minibatches",5),
    # Part 7
    ("train_step",6),("train_one_epoch",6),("train_loop",6),("evaluate",6),
]


def step_id(i):
    return f"{i:04d}"
