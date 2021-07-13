#!/usr/bin/env python3
"""Converts JSONL alignment file to CSV phoneme ids

See: https://github.com/rhasspy/kaldi-align
"""
import argparse
import json
import os
import sys

from gruut_ipa import IPA

from gruut.lang import id_to_phonemes

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(prog="align2phonemesids.py")
parser.add_argument("language", help="Language code")
parser.add_argument(
    "--no-stress", action="store_true", help="Don't include stress symbols"
)
parser.add_argument(
    "--has-speaker", action="store_true", help="CSV input has format id|speaker|text"
)
parser.add_argument(
    "--id-languages", nargs="+", help="Ordered list of languages for phoneme ids"
)

if os.isatty(sys.stdin.fileno()):
    print("Reading JSONL data from stdin...", file=sys.stderr)

args = parser.parse_args()

# -----------------------------------------------------------------------------

lang_phonemes = id_to_phonemes(args.id_languages or args.language)
phonemes_to_id = {p: i for i, p in enumerate(lang_phonemes)}

skip_phones = {"SIL", "SPN", "NSN"}

for line in sys.stdin:
    pron_obj = json.loads(line)

    split_phonemes = []
    for phoneme in pron_obj["ipa"]:
        if not phoneme:
            continue

        while phoneme and IPA.is_stress(phoneme[0]):
            split_phonemes.append(phoneme[0])
            phoneme = phoneme[1:]

        if phoneme:
            split_phonemes.append(phoneme)

    try:
        pron_ids = [phonemes_to_id[p] for p in split_phonemes if p not in skip_phones]
        print(pron_obj["id"], end="|")
        print(*pron_ids)
    except Exception as e:
        print(repr(e), line, file=sys.stderr)
