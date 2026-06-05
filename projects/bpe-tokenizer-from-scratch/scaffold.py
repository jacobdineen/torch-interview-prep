"""Build a BPE Tokenizer from Scratch — end-to-end demo.

    python projects.py bpe-tokenizer-from-scratch --scaffold

Trains a Byte-Pair-Encoding tokenizer on a small corpus, shows the merges it
learned, and round-trips text -> ids -> text. Pure Python, instant.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import *  # noqa: F401,F403


def main():
    text = ("the cat sat on the mat the cat ran the dog sat the dog ran "
            "low lower lowest newer newest wider widest")
    words = pretokenize(text)
    freqs = build_word_frequencies(words)
    syms = initial_vocab(freqs)

    merges = train_bpe(freqs, num_merges=20)
    ranks = build_merge_ranks(merges)
    tok2id = build_token_to_id(syms, merges)
    id2tok = build_id_to_token(tok2id)

    print(f"corpus: {len(words)} words, {len(syms)} base symbols")
    print(f"learned {len(merges)} merges, first 6: {[a + '+' + b for a, b in merges[:6]]}")
    print(f"vocab size: {len(tok2id)}")
    ids = encode(text, ranks, tok2id)
    back = decode(ids, id2tok)
    print(f"encoded {len(words)} words -> {len(ids)} tokens")
    print(f"round-trip exact: {back == text}")
    print("ok" if back == text and len(merges) > 0 else "warning")


if __name__ == "__main__":
    main()
