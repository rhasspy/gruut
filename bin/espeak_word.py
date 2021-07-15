#!/usr/bin/env python3
"""Generate IPA lexicon for a list of words using espeak-ng"""
import argparse
import functools
import os
import subprocess
import sys
import typing
from concurrent.futures import ThreadPoolExecutor

from gruut_ipa import Pronunciation


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="espeak_word.py")
    parser.add_argument("language", help="eSpeak voice/language")
    parser.add_argument(
        "--pos",
        action="store_true",
        help="Add pronunciations for different parts of speech (English only)",
    )
    parser.add_argument(
        "--empty-pos-tag",
        default="_",
        help="POS tag to use for default pronunciation (default: _)",
    )
    args = parser.parse_args()

    if os.isatty(sys.stdin.fileno()):
        print("Reading words from stdin...", file=sys.stderr)

    words = filter(None, map(str.strip, sys.stdin))

    with ThreadPoolExecutor() as executor:
        for word, word_prons in executor.map(
            functools.partial(phonemize_word, voice=args.language, pos=args.pos), words
        ):
            for word_pron, pos_tag in word_prons:
                if not word_pron:
                    continue

                if args.pos:
                    # word pos phonemes
                    print(word, pos_tag or args.empty_pos_tag, *word_pron)
                else:
                    # word phonemes
                    print(word, *word_pron)


# -----------------------------------------------------------------------------

WORD_PRON = typing.List[str]
PRON_AND_POS_TAG = typing.Tuple[WORD_PRON, typing.Optional[str]]
WORD_PRONS = typing.List[PRON_AND_POS_TAG]
WORD_AND_PRONS = typing.Tuple[str, WORD_PRONS]


def phonemize_word(word: str, voice: str, pos: bool = False) -> WORD_AND_PRONS:
    """Get IPA from espeak-ng for a given word/voice"""
    word_prons: WORD_PRONS = []

    ipa_str = subprocess.check_output(
        ["espeak-ng", "-q", "--ipa", "-v", voice, word], universal_newlines=True
    ).strip()

    ipa_pron = [p.text for p in Pronunciation.from_string(ipa_str)]

    # Default pronunciation
    word_prons.append((ipa_pron, None))

    if pos:
        # Use an initial word to prime eSpeak to change pronunciation for part of speech.
        # Obviously, this only works for English.
        for pos_word, pos_tag in [("preferably", "VB"), ("a", "NN"), ("had", "VBD")]:
            # Only keep last word's IPA
            pos_ipa_str = (
                subprocess.check_output(
                    ["espeak-ng", "-q", "--ipa", "-v", voice, f"{pos_word} {word}"],
                    universal_newlines=True,
                )
                .strip()
                .split()[-1]
            )

            pos_ipa_pron = [p.text for p in Pronunciation.from_string(pos_ipa_str)]
            if pos_ipa_pron != ipa_pron:
                # Only add pronunciation if it differs from the default
                word_prons.append((pos_ipa_pron, pos_tag))

    return word, word_prons


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
