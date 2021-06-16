#!/usr/bin/env python3
import csv
import json
import sys

from gruut import text_to_phonemes
from gruut.lang import id_to_phonemes
from gruut_ipa import IPA

if len(sys.argv) < 2:
    print("Usage: csv2phonemeids.py <lang> < CSV > CSV")

lang = sys.argv[1]
lang_phonemes = id_to_phonemes(lang)
phonemes_to_id = {p: i for i, p in enumerate(lang_phonemes)}

writer = csv.writer(sys.stdout, delimiter="|")
for row in csv.reader(sys.stdin, delimiter="|"):
    utt_id, text = row[0], row[1]
    phoneme_ids = []

    for phoneme in text_to_phonemes(
        text,
        lang=lang,
        return_format="flat_phonemes",
        phonemizer_args={"word_break": IPA.BREAK_WORD},
    ):
        phoneme_id = phonemes_to_id.get(phoneme)
        if phoneme_id is not None:
            phoneme_ids.append(phoneme_id)

    if phoneme_ids:
        writer.writerow((utt_id, " ".join(str(p_id) for p_id in phoneme_ids)))
