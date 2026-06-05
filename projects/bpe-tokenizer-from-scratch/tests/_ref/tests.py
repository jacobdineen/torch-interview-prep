"""Hidden tests for bpe-tokenizer-from-scratch. One test_<id>_<name>(ns) per step; deterministic, independent oracles."""
import numpy as np  # noqa: F401

def test_0001_pretokenize(ns):
    pretokenize = ns["pretokenize"]
    assert pretokenize("low low lowest") == ["low", "low", "lowest"]
    assert pretokenize("  a\tb\n c  ") == ["a", "b", "c"]
    assert pretokenize("") == []
    assert pretokenize("solo") == ["solo"]


def test_0002_word_to_symbols(ns):
    word_to_symbols = ns["word_to_symbols"]
    END = "</w>"
    assert word_to_symbols("low") == ("l", "o", "w", END)
    assert word_to_symbols("a") == ("a", END)
    # empty word -> just the marker
    assert word_to_symbols("") == (END,)
    result = word_to_symbols("ab")
    assert isinstance(result, tuple)
    assert result[-1] == END
    assert len(result) == 3


def test_0003_build_word_frequencies(ns):
    build_word_frequencies = ns["build_word_frequencies"]
    END = "</w>"
    words = ["low", "low", "lowest"]
    freqs = build_word_frequencies(words)
    expected = {
        ("l", "o", "w", END): 2,
        ("l", "o", "w", "e", "s", "t", END): 1,
    }
    assert freqs == expected
    # total occurrences preserved
    assert sum(freqs.values()) == len(words)
    # single distinct word
    assert build_word_frequencies(["a", "a", "a"]) == {("a", END): 3}


def test_0004_initial_vocab(ns):
    initial_vocab = ns["initial_vocab"]
    END = "</w>"
    word_freqs = {
        ("l", "o", "w", END): 2,
        ("l", "o", "w", "e", "s", "t", END): 1,
    }
    vocab = initial_vocab(word_freqs)
    assert vocab == {"l", "o", "w", "e", "s", "t", END}
    assert isinstance(vocab, set)
    # empty corpus -> empty set
    assert initial_vocab({}) == set()

def test_0005_count_pairs(ns):
    count_pairs = ns["count_pairs"]
    # word_freqs: ('l','o','w','</w>') x3, ('l','o','</w>') x2
    word_freqs = {
        ("l", "o", "w", "</w>"): 3,
        ("l", "o", "</w>"): 2,
    }
    got = count_pairs(word_freqs)
    # Manual oracle:
    # from low x3: (l,o),(o,w),(w,</w>) each +3
    # from lo x2:  (l,o),(o,</w>) each +2
    expected = {
        ("l", "o"): 5,
        ("o", "w"): 3,
        ("w", "</w>"): 3,
        ("o", "</w>"): 2,
    }
    assert got == expected, (got, expected)
    # empty corpus -> no pairs
    assert count_pairs({}) == {}
    # single-symbol word has no adjacent pair
    assert count_pairs({("a",): 7}) == {}


def test_0006_best_pair(ns):
    best_pair = ns["best_pair"]
    # Distinct max
    assert best_pair({("a", "b"): 5, ("c", "d"): 2}) == ("a", "b")
    # Tie on count -> lexicographically smallest pair
    counts = {("z", "z"): 4, ("a", "b"): 4, ("a", "a"): 4}
    # min lexicographic among the tied pairs is ("a","a")
    assert best_pair(counts) == ("a", "a")
    # Tie where second element decides
    assert best_pair({("a", "b"): 1, ("a", "a"): 1}) == ("a", "a")
    # Empty -> None
    assert best_pair({}) is None
    assert best_pair(None) is None


def test_0007_merge_word(ns):
    merge_word = ns["merge_word"]
    # Replace adjacent (a,b)
    assert merge_word(("a", "b", "c"), ("a", "b")) == ("ab", "c")
    # No occurrence -> unchanged
    assert merge_word(("a", "c", "b"), ("a", "b")) == ("a", "c", "b")
    # Non-overlapping left-to-right consumption: aa with pair (a,a)
    assert merge_word(("a", "a", "a"), ("a", "a")) == ("aa", "a")
    assert merge_word(("a", "a", "a", "a"), ("a", "a")) == ("aa", "aa")
    # Multiple separated occurrences
    assert merge_word(("a", "b", "x", "a", "b"), ("a", "b")) == ("ab", "x", "ab")
    # Result is a tuple
    assert isinstance(merge_word(("a", "b"), ("a", "b")), tuple)
    # Empty / single symbol untouched
    assert merge_word((), ("a", "b")) == ()
    assert merge_word(("a",), ("a", "b")) == ("a",)


def test_0008_apply_merge(ns):
    apply_merge = ns["apply_merge"]
    word_freqs = {
        ("l", "o", "w", "</w>"): 3,
        ("l", "o", "</w>"): 2,
    }
    got = apply_merge(word_freqs, ("l", "o"))
    expected = {
        ("lo", "w", "</w>"): 3,
        ("lo", "</w>"): 2,
    }
    assert got == expected, (got, expected)
    # Frequencies preserved (sum unchanged)
    assert sum(got.values()) == sum(word_freqs.values())
    # Original dict not mutated
    assert word_freqs == {
        ("l", "o", "w", "</w>"): 3,
        ("l", "o", "</w>"): 2,
    }
    # Keys that collapse to the same tuple get their freqs summed
    collide = {("a", "b", "c"): 1, ("ab", "c"): 4}
    merged = apply_merge(collide, ("a", "b"))
    assert merged == {("ab", "c"): 5}

def test_0009_train_bpe(ns):
    train_bpe = ns["train_bpe"]
    END = "</w>"

    # Crafted corpus: "ab" appears more than anything, so ('a','b') is the
    # first merge. After merging, ('ab','</w>') becomes the dominant pair.
    # words: "ab" x3, "ac" x1
    word_freqs = {
        ("a", "b", END): 3,
        ("a", "c", END): 1,
    }

    # Manual expectation for num_merges=2:
    # Iteration 0 pair counts (weighted):
    #   ('a','b'):3, ('b',</w>):3, ('a','c'):1, ('c',</w>):1
    #   best = ('a','b')  -> merge
    #   words: ("ab",</w>):3, ("a","c",</w>):1
    # Iteration 1 pair counts:
    #   ('ab',</w>):3, ('a','c'):1, ('c',</w>):1
    #   best = ('ab',</w>) -> merge
    expected = [("a", "b"), ("ab", END)]
    got = train_bpe(word_freqs, 2)
    assert got == expected, got

    # Determinism: same input -> same output
    assert train_bpe(word_freqs, 2) == got

    # Early stop: a single fully-merged word has no pairs after enough merges.
    # ("x","y",</w>) -> merges ('x','y') then ('xy',</w>) then nothing left.
    single = {("x", "y", END): 1}
    res = train_bpe(single, 10)
    assert len(res) == 2, res
    assert res == [("x", "y"), ("xy", END)], res

    # num_merges=0 -> empty list
    assert train_bpe(word_freqs, 0) == []

    # Prefix invariant: training with fewer merges yields a prefix of more merges.
    more = train_bpe(word_freqs, 2)
    fewer = train_bpe(word_freqs, 1)
    assert more[: len(fewer)] == fewer, (fewer, more)


def test_0010_build_merge_ranks(ns):
    build_merge_ranks = ns["build_merge_ranks"]

    merges = [("a", "b"), ("ab", "</w>"), ("c", "d")]
    ranks = build_merge_ranks(merges)

    # Each pair maps to its index.
    assert ranks == {("a", "b"): 0, ("ab", "</w>"): 1, ("c", "d"): 2}, ranks

    # Round-trip: index recovers the pair.
    for i, pair in enumerate(merges):
        assert ranks[pair] == i

    # Empty input -> empty mapping.
    assert build_merge_ranks([]) == {}

    # Size matches (no collisions when pairs are distinct).
    assert len(ranks) == len(merges)

def test_0011_bpe_encode_word(ns):
    bpe_encode_word = ns["bpe_encode_word"]
    END = "</w>"

    # Empty merge_ranks: word stays as raw characters + END.
    out = bpe_encode_word("cat", {})
    assert out == ["c", "a", "t", END], out

    # Hand-crafted merge ranks. Word "low" -> symbols ('l','o','w','</w>').
    # ranks: ('l','o')=0, ('o','w')=1, ('lo','w')=2, ('low','</w>')=3
    merge_ranks = {
        ("l", "o"): 0,
        ("o", "w"): 1,
        ("lo", "w"): 2,
        ("low", END): 3,
    }
    # Manual oracle:
    # ['l','o','w','</w>'] lowest rank pair = ('l','o') rank0 -> ['lo','w','</w>']
    # pairs: ('lo','w') rank2 -> ['low','</w>']
    # pairs: ('low','</w>') rank3 -> ['low</w>']
    out = bpe_encode_word("low", merge_ranks)
    assert out == ["low" + END], out

    # Word "ow" -> ('o','w','</w>'): only ('o','w') rank1 applies -> ['ow','</w>']
    out2 = bpe_encode_word("ow", merge_ranks)
    assert out2 == ["ow", END], out2

    # Determinism: same input gives same output.
    assert bpe_encode_word("low", merge_ranks) == out


def test_0012_build_token_to_id(ns):
    build_token_to_id = ns["build_token_to_id"]
    END = "</w>"

    initial = {"w", "l", "o", END}
    merges = [("l", "o"), ("lo", "w")]
    t2i = build_token_to_id(initial, merges)

    # sorted initial symbols: ['</w>', 'l', 'o', 'w'] (since '</w>' < lowercase letters)
    sorted_init = sorted(initial)
    expected = {}
    for s in sorted_init:
        expected[s] = len(expected)
    for a, b in merges:
        expected[a + b] = len(expected)

    assert t2i == expected, (t2i, expected)
    # No duplicate ids, contiguous 0..n-1.
    ids = list(t2i.values())
    assert sorted(ids) == list(range(len(t2i))), ids
    assert len(set(ids)) == len(ids)

    # Idempotent / no-merge-collision: if a merged token already exists as a symbol,
    # it is not reassigned. Construct such a case.
    init2 = {"a", "b", "ab"}
    t2i2 = build_token_to_id(init2, [("a", "b")])
    # 'ab' already present from init; merge should not create a duplicate or new id.
    assert t2i2["ab"] == sorted(init2).index("ab"), t2i2
    assert sorted(t2i2.values()) == list(range(len(t2i2)))


def test_0013_build_id_to_token(ns):
    build_id_to_token = ns["build_id_to_token"]

    t2i = {"</w>": 0, "l": 1, "o": 2, "w": 3, "lo": 4, "low": 5}
    i2t = build_id_to_token(t2i)

    # Manual inverse.
    expected = {0: "</w>", 1: "l", 2: "o", 3: "w", 4: "lo", 5: "low"}
    assert i2t == expected, i2t

    # Round-trip property: inverting twice recovers original.
    again = {tok: i for i, tok in i2t.items()}
    assert again == t2i, again

    # Every id maps back to a token whose forward id matches.
    for tok, i in t2i.items():
        assert i2t[i] == tok

def test_0014_encode(ns):
    encode = ns["encode"]
    # Corpus where we control merges by hand.
    # Words: "ab", "ab" -> symbols ('a','b','</w>') each.
    # Provide merge_ranks so that ('a','b') merges first (rank 0),
    # then ('ab','</w>') merges (rank 1).
    merge_ranks = {("a", "b"): 0, ("ab", "</w>"): 1}
    # token ids assigned deterministically (manual).
    token_to_id = {"a": 0, "b": 1, "</w>": 2, "ab": 3, "ab</w>": 4}

    # Manually encode "ab cd".
    # word "ab": symbols ('a','b','</w>')
    #   lowest-rank adjacent pair in merge_ranks: ('a','b') rank 0 -> ('ab','</w>')
    #   then ('ab','</w>') rank 1 -> ('ab</w>',) -> token "ab</w>" -> id 4
    # word "cd": symbols ('c','d','</w>') -> no merges apply
    #   tokens "c","d","</w>" -> ids must exist in token_to_id
    token_to_id_full = dict(token_to_id)
    token_to_id_full["c"] = 5
    token_to_id_full["d"] = 6

    got = encode("ab cd", merge_ranks, token_to_id_full)
    expected = [4, 5, 6, 2]
    assert got == expected, (got, expected)

    # Determinism: same call yields same result.
    again = encode("ab cd", merge_ranks, token_to_id_full)
    assert again == expected

    # Empty text -> empty id list.
    assert encode("", merge_ranks, token_to_id_full) == []


def test_0015_decode(ns):
    decode = ns["decode"]
    END = "</w>"

    # id_to_token inverse of a hand-made mapping.
    id_to_token = {0: "a", 1: "b", 2: END, 3: "ab", 4: "ab" + END, 5: "c", 6: "d"}

    # Reconstruct "ab cd": tokens for "ab" then "</w>", then "cd" then "</w>".
    # [4, 5, 6, 2] -> "ab</w>" + "c" + "d" + "</w>"
    #             -> "ab</w>cd</w>" -> replace </w> with space -> "ab cd " -> strip trailing -> "ab cd"
    ids = [4, 5, 6, 2]
    got = decode(ids, id_to_token)
    assert got == "ab cd", got

    # Manual oracle: join, replace, strip trailing space.
    raw = "".join(id_to_token[i] for i in ids)
    expected = raw.replace(END, " ")
    if expected.endswith(" "):
        expected = expected[:-1]
    assert got == expected

    # Round-trip with encode using a consistent vocab.
    encode = ns["encode"]
    merge_ranks = {("a", "b"): 0, (("a" + "b"), END): 1}
    token_to_id = {"a": 0, "b": 1, END: 2, "ab": 3, "ab" + END: 4, "c": 5, "d": 6}
    id_to_token2 = {v: k for k, v in token_to_id.items()}
    text = "ab cd"
    assert decode(encode(text, merge_ranks, token_to_id), id_to_token2) == text

    # Single word round-trips (trailing space stripped).
    assert decode([4], id_to_token) == "ab"
