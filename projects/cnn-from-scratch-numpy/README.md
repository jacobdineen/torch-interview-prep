# Build a Trainable CNN from Scratch in NumPy

Assemble a LeNet-style convolutional network entirely in **NumPy** — from a
numerically stable softmax and im2col-based convolutions all the way to an
Adam-driven training loop. Every layer, gradient, and optimizer is wired by hand
into a working classifier you can train on synthetic images. **59 steps, 7 parts.**

| Part | Focus |
|------|-------|
| 1 | Stable softmax, cross-entropy, accuracy |
| 2 | He init, padding, output-size math, im2col / col2im |
| 3 | Conv / maxpool / ReLU / flatten / linear forward **and backward** |
| 4 | Fused softmax-CE, SGD, and Adam (step by step) |
| 5 | Assembling LeNet: blocks, full forward/backward, predict |
| 6 | Synthetic image data, shuffling, split, minibatches |
| 7 | Training step, epoch loop, driver, evaluation |

Backward passes are checked against **finite-difference gradients**.

```bash
uv run python projects.py cnn-from-scratch-numpy            # parts + steps
uv run python projects.py cnn-from-scratch-numpy --next     # next unsolved step
uv run python projects.py cnn-from-scratch-numpy --scaffold # end-to-end demo (once solved)
```
