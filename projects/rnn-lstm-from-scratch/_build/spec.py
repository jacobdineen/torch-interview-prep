"""Build spec for the rnn-lstm-from-scratch project."""

TITLE = 'Build an RNN and LSTM from Scratch'

PARTS = [
    ("Character Data", "Build a character vocabulary, encode text to ids, make next-char training windows, and one-hot encode."),
    ("Vanilla RNN", "Implement the recurrent cell, unroll it over a sequence, and project hidden states to vocab logits."),
    ("LSTM", "Implement the gated LSTM cell (input/forget/output/candidate), unroll it, and project to logits."),
    ("Loss & Training", "Sequence cross-entropy, the loss over a model's logits, an SGD step (torch autograd), and the training loop."),
    ("Sampling & Evaluation", "Sample the next character with temperature, autoregressively generate text, and measure perplexity."),
]

STEPS = [
    ("build_char_vocab", 0), ("encode_text", 0), ("make_training_pairs", 0), ("one_hot", 0),
    ("init_rnn_params", 1), ("rnn_cell_step", 1), ("rnn_forward", 1), ("output_logits", 1), ("rnn_logits", 1),
    ("init_lstm_params", 2), ("lstm_gates", 2), ("lstm_cell_step", 2), ("lstm_forward", 2), ("lstm_logits", 2),
    ("sequence_cross_entropy", 3), ("compute_sequence_loss", 3), ("train_step", 3), ("train_model", 3),
    ("sample_next_char", 4), ("generate_text", 4), ("sequence_perplexity", 4),
]


def step_id(i):
    return f"{i:04d}"
