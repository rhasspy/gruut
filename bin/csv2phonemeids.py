#!/usr/bin/env python3
"""Convert CSV metadata file with text to CSV phoneme ids"""
import argparse
import csv
import os
import sys

from gruut import text_to_phonemes
from gruut.lang import id_to_phonemes
from gruut_ipa import IPA

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(prog="csv2phonemeids.py")
parser.add_argument("lang", help="Language code")
parser.add_argument(
    "--no-stress", action="store_true", help="Don't include stress symbols"
)
parser.add_argument(
    "--has-speaker", action="store_true", help="CSV input has format id|speaker|text"
)

if os.isatty(sys.stdin.fileno()):
    print("Reading CSV data from stdin...", file=sys.stderr)

args = parser.parse_args()

# -----------------------------------------------------------------------------

lang_phonemes = id_to_phonemes(args.lang, no_stress=args.no_stress)
phonemes_to_id = {p: i for i, p in enumerate(lang_phonemes)}

writer = csv.writer(sys.stdout, delimiter="|")
for row in csv.reader(sys.stdin, delimiter="|"):
    if args.has_speaker:
        utt_id, _speaker, text = row[0], row[1], row[2]
    else:
        utt_id, text = row[0], row[1]

    phoneme_ids = []

    for phoneme in text_to_phonemes(
        text,
        lang=args.lang,
        return_format="flat_phonemes",
        phonemizer_args={"word_break": IPA.BREAK_WORD},
    ):
        assert isinstance(phoneme, str)
        phoneme_id = phonemes_to_id.get(phoneme)
        if phoneme_id is not None:
            phoneme_ids.append(phoneme_id)

    if phoneme_ids:
        writer.writerow((utt_id, " ".join(str(p_id) for p_id in phoneme_ids)))
