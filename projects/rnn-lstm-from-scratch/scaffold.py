"""Build an RNN and LSTM from Scratch — char-level language model demo.

    python projects.py rnn-lstm-from-scratch --scaffold

Trains a vanilla RNN and an LSTM (same code path, swap the forward fn) on a tiny
character corpus with torch autograd, then samples text from the trained LSTM.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import *  # noqa: F401,F403


def main():
    text = "hello world. " * 40
    stoi, itos = build_char_vocab(text)
    V = len(stoi)
    ids = encode_text(text, stoi)
    X_ids, Y = make_training_pairs(ids, seq_len=12)
    X = one_hot(X_ids, V)
    print(f"corpus: {len(text)} chars, vocab V={V}, {X.shape[0]} windows of len {X.shape[1]}")

    for label, init, fwd, args in [
        ("RNN", init_rnn_params, rnn_logits, (V, 32)),
        ("LSTM", init_lstm_params, lstm_logits, (V, 32)),
    ]:
        params = init(*args, seed=0)
        hist = train_model(params, X, Y, lr=0.5, n_steps=1200, forward_fn=fwd)
        ppl = sequence_perplexity(params, X, Y, fwd)
        print(f"{label}: loss {hist[0]:.3f} -> {hist[-1]:.3f}   perplexity {ppl:.2f}")
        if label == "LSTM":
            g = torch.Generator().manual_seed(0)
            out = generate_text(params, stoi, itos, "hello", 20, fwd, temperature=0.5, generator=g)
            print(f"  sample: {out!r}")
            ok = hist[-1] < hist[0] * 0.6
    print("ok" if ok else "warning: weak fit")


if __name__ == "__main__":
    main()
