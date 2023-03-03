#!/usr/bin/env python3
"""Compute PER (phoneme error rate) on a lexicon using Phonetisaurus G2P FST"""
import argparse
import itertools
import logging
import os
import sys
import time
import typing

_LOGGER = logging.getLogger("phonetisaurus_per")

logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("fst", help="Path to Phonetisaurus g2p fst model")
parser.add_argument(
    "texts", nargs="*", help="Lines with '<word> <phoneme> <phoneme> ...'"
)
parser.add_argument(
    "--batch-size", type=int, default=1000, help="Number of words to run at a time"
)
args = parser.parse_args()

# -----------------------------------------------------------------------------

try:
    import phonetisaurus
except ImportError as e:
    _LOGGER.critical("phonetisaurus library is needed for g2p")
    _LOGGER.critical("pip install 'phonentisaurus>=0.3.0'")
    raise e

try:
    from rapidfuzz.distance.Levenshtein import distance as levenshtein
except ImportError as e:
    _LOGGER.critical("rapidfuzz library is needed for levenshtein distance")
    _LOGGER.critical("pip install 'rapidfuzz>=2.11.1'")
    raise e


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    zip_args = [iter(iterable)] * n
    return itertools.zip_longest(*zip_args, fillvalue=fillvalue)


# -----------------------------------------------------------------------------

# Load lexicon
if args.texts:
    lines = args.texts
else:
    lines = sys.stdin

    if os.isatty(sys.stdin.fileno()):
        print("Reading lexicon lines from stdin...", file=sys.stderr)

lexicon = {}
for line in lines:
    line = line.strip()
    if (not line) or (" " not in line):
        continue

    word, actual_phonemes = line.split(maxsplit=1)
    lexicon[word] = actual_phonemes

# Predict phonemes
predicted_phonemes: typing.Dict[str, str] = {}
start_time = time.perf_counter()

for words in grouper(lexicon.keys(), args.batch_size):
    words = filter(None, words)
    predicted_phonemes.update(
        (word, " ".join(phonemes))
        for word, phonemes in phonetisaurus.predict(words=words, model_path=args.fst)
    )

end_time = time.perf_counter()

# Calculate PER
num_errors = 0
num_missing = 0
num_phonemes = 0

for word, actual_phonemes in lexicon.items():
    expected_phonemes = predicted_phonemes.get(word, "")
    if expected_phonemes:
        distance = levenshtein(expected_phonemes, actual_phonemes)
        num_errors += distance
        num_phonemes += len(actual_phonemes)
    else:
        num_missing += 1
        _LOGGER.warning("No pronunciation for %s", word)

    # print(word, actual_phonemes, expected_phonemes, sep=" | ")

assert num_phonemes > 0, "No phonemes were read"

# Calculate results
per = round(num_errors / num_phonemes, 2)
wps = round(len(predicted_phonemes) / (end_time - start_time), 2)
print("PER:", per, "Errors:", num_errors, "words/sec:", wps)

if num_missing > 0:
    print("Total missing:", num_missing)
