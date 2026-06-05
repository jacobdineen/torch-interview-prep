"""Attention Is All You Need — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py attention-is-all-you-need --scaffold

Assembles your step functions into a tiny encoder-decoder Transformer and trains
it on a toy COPY task (the target is the source sequence) with the Noam schedule
+ label-smoothed loss + your from-scratch Adam, then greedily decodes a sample.
Loss should fall and the model should learn to copy. Pure CPU, well under a minute.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def main():
    torch.manual_seed(0)
    d_model, n_heads, d_ff, n_layers = 32, 4, 64, 2
    vocab = 12          # ids 4..11 are content tokens; 0=pad 1=bos 2=eos
    B, S = 8, 6

    # toy copy task: target == source (content ids 4..vocab-1), eos appended
    src = torch.randint(4, vocab, (B, S))
    tgt = src.clone()

    encoder = [init_encoder_layer_parameters(d_model, d_ff, n_heads) for _ in range(n_layers)]
    decoder = [init_decoder_layer_parameters(d_model, d_ff, n_heads) for _ in range(n_layers)]
    embed, out_bias = init_embedding_and_projection_parameters(vocab, d_model)
    params = {"d_model": d_model, "n_heads": n_heads, "vocab_size": vocab, "d_ff": d_ff,
              "n_layers": n_layers, "embed": embed, "out_bias": out_bias,
              "encoder": encoder, "decoder": decoder}

    print(f"model: d_model={d_model} heads={n_heads} layers={n_layers} vocab={vocab}")
    history = run_training_loop_for_steps(params, src, tgt, n_steps=200,
                                          d_model=d_model, warmup=40, smoothing=0.1)
    print(f"loss: {history[0]:.3f} -> {history[-1]:.3f}")

    # greedy decode the first example from BOS
    with torch.no_grad():
        dec_in = torch.full((1, 1), 1, dtype=torch.long)   # BOS
        for _ in range(S):
            logp = run_transformer_forward(params, src[:1], dec_in)
            nxt = pick_next_token_by_argmax(logp[:, -1])
            dec_in = torch.cat([dec_in, nxt.view(1, 1)], dim=1)
    print("source :", src[0].tolist())
    print("decoded:", dec_in[0, 1:].tolist())
    print("ok" if history[-1] < history[0] else "warning: loss did not fall")


if __name__ == "__main__":
    main()
