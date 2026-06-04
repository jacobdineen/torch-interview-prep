"""Tensor Ops with einops — end-to-end demo.

Run once you've solved (enough of) the steps:

    python projects.py einops --scaffold

It assembles your step functions into two tiny pipelines — a single-block
self-attention forward pass and an image-to-feature reducer — and prints the
shape at each stage so you can see the einops ops composing. Pure NumPy, runs in
well under a second on CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def attention_demo():
    """One head of scaled dot-product self-attention, built from the steps."""
    rng = np.random.default_rng(0)
    batch, seq, dim, heads = 2, 4, 8, 2
    x = rng.standard_normal((batch, seq, dim))

    q = split_into_heads(x, heads)            # b h s d
    k = split_into_heads(x, heads)
    v = split_into_heads(x, heads)
    print(f"  q/k/v per head:    {q.shape}")

    # scores per head, then a softmax, then a weighted sum of values
    scores = np.stack([attention_scores(q[:, h], k[:, h]) for h in range(heads)], axis=1)
    scores = scores / np.sqrt(q.shape[-1])
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights = weights / weights.sum(-1, keepdims=True)
    print(f"  attention weights: {weights.shape}")

    out = np.stack([weighted_token_sum(weights[:, h], v[:, h]) for h in range(heads)], axis=1)
    print(f"  per-head context:  {out.shape}")
    merged = merge_batch_and_time(out.transpose(0, 2, 1, 3).reshape(batch, seq, dim))
    print(f"  merged (b*s, dim): {merged.shape}")


def image_demo():
    """Grayscale image -> RGB -> upsample -> pool -> per-channel feature vector."""
    rng = np.random.default_rng(1)
    img = rng.standard_normal((8, 8))
    rgb = gray_to_rgb(img)                                  # h w 3
    big = upsample_nearest(img, 2)                          # 16 16
    print(f"  gray {img.shape} -> rgb {rgb.shape}, upsampled {big.shape}")

    batch = channels_last_to_first(rgb[None])              # 1 3 8 8
    pooled = max_pool_2x2(batch)                            # 1 3 4 4
    feat = global_average_pool(pooled)                     # 1 3
    print(f"  NCHW {batch.shape} -> maxpool {pooled.shape} -> feature {feat.shape}")


def main():
    print("self-attention pipeline:")
    attention_demo()
    print("image pipeline:")
    image_demo()
    print("ok")


if __name__ == "__main__":
    main()
