"""Tiny GPT from scratch — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py tiny-gpt-from-scratch --scaffold

It imports your assembled solution.py, builds a tiny GPT, trains it on a toy
corpus, reports validation loss, and samples text. It exercises the whole
pipeline: tokenizer -> data -> embeddings -> Transformer blocks -> training
(Adam) -> generation.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from solution import *  # noqa: E402,F401,F403


TOY_CORPUS = (
    "hello world\n"
    "the quick brown fox jumps over the lazy dog\n"
    "tiny gpt learns characters one step at a time\n"
) * 40


def build_model(vocab_size, block_size, d_model=16, n_heads=2, d_ff=32, n_layers=2):
    """Assemble model parameters from the step functions."""
    return {
        "tok_emb": create_token_embedding(vocab_size, d_model),
        "pos_emb": create_positional_embedding(block_size, d_model),
        "blocks": stack_transformer_blocks(n_layers, d_model, n_heads, d_ff),
        "ln_f": {"gamma": np.ones(d_model), "beta": np.zeros(d_model)},
        "lm_head": {"w_lm": np.random.randn(d_model, vocab_size) * 0.02,
                    "b_lm": np.zeros(vocab_size)},
        "block_size": block_size, "vocab_size": vocab_size,
    }


def main():
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # 1) Tokenizer + corpus
    text = read_text_file(TOY_CORPUS)
    vocab = build_vocab(text)
    stoi, itos = build_stoi(vocab), build_itos(vocab)
    vocab_size = len(vocab)
    data = encode_corpus_to_int_array(text, stoi)
    split = pick_split_point(len(data), 0.9)
    train_ids, val_ids = slice_train_and_val(data, split)
    print(f"vocab={vocab_size}  train={len(train_ids)}  val={len(val_ids)}")

    # 2) Model
    block_size = pick_block_size(16)
    params = build_model(vocab_size, block_size, d_model=16, n_heads=2, d_ff=32, n_layers=2)

    # 3) Train
    opt_state = None
    n_steps = 150
    for i in range(n_steps):
        x, y = get_batch(train_ids, block_size, 16, rng)
        params, opt_state, loss = wire_full_training_loop(params, opt_state, x, y, lr=3e-3)
        if i % 30 == 0 or i == n_steps - 1:
            print(f"  step {i:4d}  train loss {loss:.4f}")

    # 4) Validation + generation
    val_loss = logging_and_validation_loss(params, val_ids, block_size, 8, 4)
    print(f"val loss ~ {val_loss:.4f}")
    prompt = encode_prompt("the ", stoi)
    gen = generation_loop_for_n_steps(params, prompt, 80, block_size,
                                      temperature=0.8, top_k=10, rng=rng)
    print("generated:", repr(decode_final_sequence(gen, itos)))


if __name__ == "__main__":
    main()
