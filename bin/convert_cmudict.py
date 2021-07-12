#!/usr/bin/env python3
"""Converts CMUDict lexicon to gruut IPA phonemes"""
import argparse
import os
import sys

GRUUT_PHONEMES = set(
    [
        "ɑ",
        "æ",
        "ɛ",
        "ɪ",
        "ɔ",
        "ʊ",
        "ʌ",
        "i",
        "u",
        "ə",
        "ɚ",
        "eɪ",
        "aɪ",
        "oʊ",
        "ɔɪ",
        "aʊ",
        "p",
        "b",
        "t",
        "d",
        "k",
        "ɡ",
        "t͡ʃ",
        "d͡ʒ",
        "f",
        "v",
        "θ",
        "ð",
        "s",
        "z",
        "ʃ",
        "ʒ",
        "h",
        "l",
        "m",
        "n",
        "ŋ",
        "ɹ",
        "w",
        "j",
    ]
)

# https://github.com/rhdunn/amepd
# https://en.wikipedia.org/wiki/ARPABET
ARPABET = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AX": "ə",
    "AXR": "ɚ",
    "AY": "aɪ",
    "B": "b",
    "CH": "t͡ʃ",
    "D": "d",
    "DH": "ð",
    "DX": "ɾ",
    "EH": "ɛ",
    "ER": "ɚ",  # changed from ɝ
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "H": "h",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "d͡ʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "SH": "ʃ",
    "S": "s",
    "TH": "θ",
    "T": "t",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "ZH": "ʒ",
    "Z": "z",
}

UD_POS = {
    "det": "DT",
    "verb": "VB",
    "verb@past": "VBD",
    "prep": "IN",
    "noun": "NN",
    "num": "CD",
    "adj": "JJ",
    "adv": "RB",
    "pron": "PRP",
    "intj": "_",
    "conj": "_",
    "adj@attr": "JJ",
    "adj@pred": "JJ",
}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="convert_cmudict.py")
    parser.add_argument("--pos", action="store_true", help="Include part of speech")
    args = parser.parse_args()

    if os.isatty(sys.stdin.fileno()):
        print("Reading dictionary from stdin...", file=sys.stderr)

    for line_index, line in enumerate(sys.stdin):
        line = line.strip()

        # Skip blank lines and comments
        if (not line) or line.startswith(";"):
            continue

        # Remove line comment
        line, *_comment = line.split("#", maxsplit=1)

        # Split word/phonemes
        word, *phonemes = line.split()

        # Remove index
        word, *maybe_index = word.split("(", maxsplit=1)
        assert word, f"Blank word at line {line_index + 1}"
        pos = "_"

        if maybe_index:
            # Strip off final ')'
            index = "".join(maybe_index)[:-1]
            if not str.isdigit(index):
                pos = UD_POS.get(index, index)

        word = word.lower()

        ipa_phonemes = []
        for phoneme in phonemes:
            if phoneme == "-":
                # Skip syllable marker
                continue

            # Extract stress
            stress = ""
            if phoneme[-1] == "0":
                # No stress
                phoneme = phoneme[:-1]
            elif phoneme[-1] == "1":
                # Primary stress
                phoneme = phoneme[:-1]
                stress = "\u02C8"
            elif phoneme[-1] == "2":
                # Secondary stress
                phoneme = phoneme[:-1]
                stress = "\u02CC"

            ipa_phoneme = ARPABET.get(phoneme)
            assert (
                ipa_phoneme
            ), f"Missing IPA for {phoneme} ({word}, line {line_index + 1})"

            assert (
                ipa_phoneme in GRUUT_PHONEMES
            ), f"Not a gruut IPA phoneme: {ipa_phoneme} ({phoneme}, {word}, line {line_index + 1})"

            ipa_phonemes.append(stress + ipa_phoneme)

        if args.pos:
            # With part of speech
            print(word, pos, *ipa_phonemes)
        else:
            # Without part of speech
            print(word, *ipa_phonemes)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
