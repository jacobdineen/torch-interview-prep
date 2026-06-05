"""Build a Trainable CNN from Scratch in NumPy — end-to-end demo.

    python projects.py cnn-from-scratch-numpy --scaffold

Builds a small LeNet from your step functions, generates a synthetic image
dataset, trains it with your from-scratch Adam, and reports held-out accuracy.
Pure NumPy/CPU, well under a minute.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from solution import *  # noqa: E402,F401,F403


def main():
    rng = np.random.default_rng(0)
    K, size = 3, 12
    X, y = build_synthetic_image_dataset(360, size, K, rng)
    Xtr, ytr, Xte, yte = train_test_split(X, y, 0.25)
    params = init_lenet(1, K, size, rng)

    print(f"LeNet on {size}x{size} synthetic images, {K} classes  (train {len(Xtr)} / test {len(Xte)})")
    params, history = train_loop(params, Xtr, ytr, n_epochs=8, batch_size=32, lr=2e-3, rng=rng)
    h = history["loss"] if isinstance(history, dict) else history
    te_loss, te_acc = evaluate(params, Xte, yte)
    print(f"train loss: {float(h[0]):.3f} -> {float(h[-1]):.3f}   test accuracy: {te_acc:.3f}")
    print("ok" if te_acc > 0.7 else "warning: low accuracy")


if __name__ == "__main__":
    main()
