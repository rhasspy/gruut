#!/usr/bin/env python3
"""Print id/phoneme list for language"""
import argparse

from gruut.lang import id_to_phonemes

parser = argparse.ArgumentParser(prog="print_phonemeids.py")
parser.add_argument("language", nargs="+", help="Gruut language")
parser.add_argument(
    "--no-stress", action="store_true", help="Don't include stress symbols"
)
args = parser.parse_args()

lang_phonemes = id_to_phonemes(args.language, no_stress=args.no_stress)

for i, p in enumerate(lang_phonemes):
    print(i, p)
