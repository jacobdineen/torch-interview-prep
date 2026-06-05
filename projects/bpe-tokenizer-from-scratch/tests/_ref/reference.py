"""Hidden reference implementations for bpe-tokenizer-from-scratch. Steps call earlier references
(shared namespace). One def per step with a one-line docstring."""
import numpy as np  # noqa: F401

END = "</w>"


def pretokenize(text):
    """Split text on whitespace into a list of non-empty word strings."""
    return text.split()


def word_to_symbols(word):
    """Return the word's characters followed by the END marker as a tuple."""
    return tuple(word) + (END,)


def build_word_frequencies(words):
    """Map each word's symbol-tuple to the number of times that word occurs."""
    freqs = {}
    for word in words:
        symbols = word_to_symbols(word)
        freqs[symbols] = freqs.get(symbols, 0) + 1
    return freqs


def initial_vocab(word_freqs):
    """Return the set of all distinct symbols appearing across the keys."""
    vocab = set()
    for symbols in word_freqs:
        vocab.update(symbols)
    return vocab

END = "</w>"


def count_pairs(word_freqs):
    """Count every adjacent symbol pair, weighted by each word's frequency."""
    pair_counts = {}
    for symbols, freq in word_freqs.items():
        for a, b in zip(symbols, symbols[1:]):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + freq
    return pair_counts


def best_pair(pair_counts):
    """Return the highest-count pair, breaking ties by lexicographically smallest pair.

    Return None if pair_counts is empty or None (treat a falsy argument as having no pairs).
    """
    if not pair_counts:
        return None
    return min(pair_counts, key=lambda p: (-pair_counts[p], p))


def merge_word(symbols, pair):
    """Replace each adjacent occurrence of pair=(a,b) in the tuple with the merged symbol a+b.

    Scan left to right, consuming matches non-overlapping: when symbols[i]==a and symbols[i+1]==b emit the merged symbol a+b and advance by 2, else emit symbols[i] and advance by 1. So ('a','a','a') -> ('aa','a'). Return a tuple.
    """
    a, b = pair
    merged = a + b
    out = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def apply_merge(word_freqs, pair):
    """Return a new word_freqs with pair merged in every key, preserving frequencies.

    If two different keys collapse to the same merged tuple, sum their frequencies into one entry. Build and return a NEW dict; do not mutate the input.
    """
    new_freqs = {}
    for symbols, freq in word_freqs.items():
        new_key = merge_word(symbols, pair)
        new_freqs[new_key] = new_freqs.get(new_key, 0) + freq
    return new_freqs

def train_bpe(word_freqs, num_merges):
    """Return the ordered list of merge pairs learned by greedy BPE over the corpus."""
    merges = []
    freqs = dict(word_freqs)
    for _ in range(num_merges):
        pair_counts = count_pairs(freqs)
        pair = best_pair(pair_counts)
        if pair is None:
            break
        merges.append(pair)
        freqs = apply_merge(freqs, pair)
    return merges


def build_merge_ranks(merges):
    """Map each merge pair to its position (rank) in the learned merge list."""
    return {pair: i for i, pair in enumerate(merges)}

def bpe_encode_word(word, merge_ranks):
    """Greedily merge the word's symbols by repeatedly applying the lowest-rank adjacent merge until none remain.

    Return a LIST (not a tuple) of token strings; with empty merge_ranks this is just the word's characters plus the '</w>' marker. When several adjacent pairs are mergeable, apply the one with the lowest rank (earliest learned) first, repeating until no adjacent pair appears in merge_ranks.
    """
    symbols = list(word_to_symbols(word))
    while len(symbols) >= 2:
        best_rank = None
        best_idx = None
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            rank = merge_ranks.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_idx = i
        if best_idx is None:
            break
        merged = symbols[best_idx] + symbols[best_idx + 1]
        symbols = symbols[:best_idx] + [merged] + symbols[best_idx + 2:]
    return symbols


def build_token_to_id(initial_symbols, merges):
    """Assign ids to the sorted initial symbols, then to each merged token in merge order, with no duplicates."""
    token_to_id = {}
    for sym in sorted(initial_symbols):
        if sym not in token_to_id:
            token_to_id[sym] = len(token_to_id)
    for a, b in merges:
        tok = a + b
        if tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)
    return token_to_id


def build_id_to_token(token_to_id):
    """Return the inverse mapping from id back to token string."""
    return {i: tok for tok, i in token_to_id.items()}

def encode(text, merge_ranks, token_to_id):
    """Tokenize text into BPE tokens and map them to their integer ids."""
    ids = []
    for word in pretokenize(text):
        for token in bpe_encode_word(word, merge_ranks):
            ids.append(token_to_id[token])
    return ids


def decode(ids, id_to_token):
    """Map ids back to tokens and reconstruct the original whitespace-joined text."""
    text = "".join(id_to_token[i] for i in ids)
    text = text.replace(END, " ")
    if text.endswith(" "):
        text = text[:-1]
    return text
