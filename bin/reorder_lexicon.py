#!/usr/bin/env python3
"""Re-orders lexicon entries based on pronunciation frequencies found in a JSONL alignment file.

See: https://github.com/rhasspy/kaldi-align
"""
import json
import sys
import typing
from collections import Counter, defaultdict

SKIP_WORDS = {"<eps>", "<unk>"}

if len(sys.argv) < 2:
    print(
        "Usage: reorder_lexicon.py aligned_phones.jsonl [phoneme_col=1] < lexicon.txt > reordered_lexicon.txt"
    )
    sys.exit(1)

aligned_path = sys.argv[1]
phoneme_col = 1

if len(sys.argv) > 2:
    phoneme_col = int(sys.argv[2])


lexicon: typing.Dict[str, typing.Dict[typing.Tuple[str, ...], str]] = defaultdict(dict)
num_prons: typing.Counter[str] = Counter()

# word -> pron counts
pron_counts: typing.Dict[str, typing.Counter[typing.Tuple[str, ...]]] = defaultdict(
    Counter
)

# Load lexicon
print("Loading lexicon...", file=sys.stderr)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    parts = line.split()
    assert len(parts) > 1, line

    word = parts[0]
    phonemes = tuple(parts[phoneme_col:])

    lexicon[word][phonemes] = line

    num_prons[word] += 1

print("Loading alignments...", file=sys.stderr)
with open(aligned_path, "r", encoding="utf-8") as aligned_file:
    for line in aligned_file:
        line = line.strip()
        if not line:
            continue

        alignment = json.loads(line)
        for word_pron in alignment["prons"]:
            word = word_pron["word"]
            if (word in SKIP_WORDS) or (num_prons[word] < 1):
                continue

            phonemes = tuple(word_pron["phones"])
            pron_counts[word][phonemes] += 1


print("Re-ordering lexicon...", file=sys.stderr)
for word in sorted(lexicon.keys()):
    # pylint: disable=cell-var-from-loop
    word_pron_counts = pron_counts[word]
    phonemes_lines = sorted(
        lexicon[word].items(),
        key=lambda kv: word_pron_counts.get(kv[0], 0),
        reverse=True,
    )

    for _, line in phonemes_lines:
        print(line)

print("Done", file=sys.stderr)
