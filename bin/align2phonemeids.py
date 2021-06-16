#!/usr/bin/env python3
import json
import sys

from gruut.lang import id_to_phonemes
from gruut_ipa import IPA

if len(sys.argv) < 2:
    print("Usage: align2phonemeids.py <lang> < JSONL > CSV")

lang = sys.argv[1]
lang_phonemes = id_to_phonemes(lang)
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
