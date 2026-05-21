"""
Problem 64: Bidirectional LSTM with Padded/Packed Sequences

A standard recipe for variable-length sequence encoding:

  - Pad the sequences (or take a padded batch as input).
  - pack_padded_sequence using actual lengths.
  - Run a bidirectional LSTM on the packed input.
  - pad_packed_sequence to recover an (B, T, 2*H) tensor.
  - Concat the final forward and backward states to produce a sentence embedding.

Implement:

  - encode_packed(lstm, padded, lengths)
        * lstm: an nn.LSTM(input_size=D, hidden_size=H, bidirectional=True, batch_first=True)
        * padded: (B, T, D) FloatTensor with right-padded zeros
        * lengths: (B,) LongTensor of unpadded lengths (must be > 0)
        Returns:
            outputs: (B, T, 2*H) padded outputs (padding positions are zeros).
            sentence_emb: (B, 2*H) where the first H dims are the final forward state
                          (h_n[0]) and the last H dims are the final backward state
                          (h_n[1]). Returns a single sentence embedding per example.
"""

import torch
import torch.nn as nn


def encode_packed(lstm, padded, lengths):
    raise NotImplementedError



if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from runner import run_test_for
    raise SystemExit(run_test_for(os.path.abspath(__file__)))
