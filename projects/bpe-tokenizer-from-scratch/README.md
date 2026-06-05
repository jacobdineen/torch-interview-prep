# Build a BPE Tokenizer from Scratch

Train a real **Byte-Pair-Encoding** tokenizer from first principles: pre-tokenize
text, represent words as symbol sequences with an end-of-word marker, count
adjacent pair frequencies, greedily merge the most frequent pair, repeat to learn
an ordered merge table, then encode text to token ids and decode back. **15 steps,
5 parts.**

| Part | Focus |
|------|-------|
| 1 | Corpus & symbols (pre-tokenize, end-of-word marker, frequencies) |
| 2 | Pair statistics & merging |
| 3 | Training the ordered merge list + rank table |
| 4 | Greedy tokenization + the token→id vocab |
| 5 | Encode text→ids and decode ids→text (exact round-trip) |

```bash
uv run python projects.py bpe-tokenizer-from-scratch            # parts + steps
uv run python projects.py bpe-tokenizer-from-scratch --scaffold # train + round-trip demo
```
