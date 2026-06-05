"""Build spec for the bpe-tokenizer-from-scratch project."""

TITLE = 'Build a BPE Tokenizer from Scratch'

PARTS = [
    ("Corpus & Symbols", "Pre-tokenize text into words and represent each as a sequence of symbols with an end-of-word marker."),
    ("Pair Statistics & Merging", "Count adjacent symbol-pair frequencies, find the most frequent pair, and merge it across the corpus."),
    ("Training the Merges", "Learn an ordered list of merges and build the rank table that drives encoding."),
    ("Tokenizing & Vocab", "Greedily apply learned merges to a word and assign integer ids to every token."),
    ("Encode & Decode", "Turn text into token ids and reconstruct the original text from ids."),
]

STEPS = [
    ("pretokenize", 0), ("word_to_symbols", 0), ("build_word_frequencies", 0), ("initial_vocab", 0),
    ("count_pairs", 1), ("best_pair", 1), ("merge_word", 1), ("apply_merge", 1),
    ("train_bpe", 2), ("build_merge_ranks", 2),
    ("bpe_encode_word", 3), ("build_token_to_id", 3), ("build_id_to_token", 3),
    ("encode", 4), ("decode", 4),
]


def step_id(i):
    return f"{i:04d}"
