#!/usr/bin/env python3
"""Normalizes lexicon using gruut phonemes"""
import logging
import os
import sys
import typing
from collections import Counter

from gruut_ipa import Phonemes

_LOGGER = logging.getLogger("phonemize_lexicon")

if len(sys.argv) < 2:
    print("Usage: phonemize_lexicon.py LANG < LEXICON > LEXICON")
    sys.exit(1)

# -----------------------------------------------------------------------------

logging.basicConfig()

lang = sys.argv[1]
phonemes = Phonemes.from_language(lang)

if os.isatty(sys.stdin.fileno()):
    print("Reading lexicon from stdin...")

unknown_counts: typing.Counter[str] = Counter()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    word, word_pron_str = line.split(maxsplit=1)

    pron_phonemes = phonemes.split(word_pron_str)
    pron_phonemes_str = " ".join(p.text for p in pron_phonemes).strip()

    if not pron_phonemes_str:
        # Don't print words with empty phonemic pronunciations
        _LOGGER.warning("No pronunciation for '%s': %s", word, word_pron_str)
        continue

    # Drop words with unknown phonemes
    unknown = []
    for phoneme in pron_phonemes:
        if phoneme.unknown:
            unknown_counts[phoneme.text] += 1
            unknown.append(phoneme.text)

    if unknown:
        _LOGGER.warning("Unknown phonemes in '%s': %s", word, unknown)
        continue

    print(word, pron_phonemes_str)

if unknown_counts:
    _LOGGER.warning("%s unknown phonemes:", len(unknown_counts))
    _LOGGER.warning(unknown_counts.most_common())
