#!/usr/bin/env python3
"""Generate IPA lexicon for a list of words using espeak-ng"""
import argparse
import functools
import itertools
import logging
import os
import subprocess
import sys
import typing
from concurrent.futures import ThreadPoolExecutor

from gruut_ipa import Pronunciation

_LOGGER = logging.getLogger("espeak_word")


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of words to process at a time per thread",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if os.isatty(sys.stdin.fileno()):
        print("Reading words from stdin...", file=sys.stderr)

    words = filter(None, map(str.strip, sys.stdin))
    word_groups = grouper(words, args.batch_size)

    with ThreadPoolExecutor() as executor:
        for results in executor.map(
            functools.partial(phonemize_words, voice=args.language, pos=args.pos),
            word_groups,
        ):
            for word, word_pron, pos_tag in results:
                if not word_pron:
                    continue

                if args.pos:
                    # word pos phonemes
                    print(word, pos_tag or args.empty_pos_tag, word_pron)
                else:
                    # word phonemes
                    print(word, word_pron)


# -----------------------------------------------------------------------------

WORD = str
WORD_PRON = str
POS_TAG = typing.Optional[str]


def phonemize_words(
    words: typing.Iterable[str], voice: str, pos: bool = False
) -> typing.Iterable[typing.Tuple[WORD, WORD_PRON, POS_TAG]]:
    """Get IPA from espeak-ng for a given word/voice"""
    # Drop empty words
    words = filter(None, words)

    # prompt, POS tag
    prompt_pos: typing.List[typing.Tuple[str, POS_TAG]] = [("", None)]

    if pos:
        # Use an initial word to prime eSpeak to change pronunciation for part of speech.
        # Obviously, this only works for English.
        prompt_pos.extend([("preferably", "VB"), ("a", "NN"), ("had", "VBD")])

    words_and_prompts = []
    for word in words:
        for prompt, pos_tag in prompt_pos:
            words_and_prompts.append((prompt, word, pos_tag))

    def espeak_input(item):
        prompt, word, _pos_tag = item
        if prompt:
            return f"{prompt} {word}"

        return word

    def espeak_output(line):
        # Return last pronunciation
        line = line.strip()
        if line:
            return line.split()[-1]

        return line

    # Process words in batch
    ipa_strs = [
        espeak_output(ipa_str)
        for ipa_str in subprocess.run(
            ["espeak-ng", "-q", "--ipa", "-v", voice],
            input="\n".join(map(espeak_input, words_and_prompts)),
            capture_output=True,
            universal_newlines=True,
            check=True,
        ).stdout.splitlines()
    ]

    if len(words_and_prompts) == len(ipa_strs):
        ipa_prons = map(Pronunciation.from_string, ipa_strs)

        for (_prompt, word, pos_tag), ipa_pron in zip(words_and_prompts, ipa_prons):
            yield word, ipa_pron.text, pos_tag
    else:
        # Fall back to word-by-word
        _LOGGER.warning(
            "Missing pronunciations from eSpeak. Falling back to slower word-by-word method."
        )

        for item in words_and_prompts:
            input_line = espeak_input(item)
            prompt, word, pos_tag = item

            ipa_str = espeak_output(
                subprocess.check_output(
                    ["espeak-ng", "-q", "--ipa", "-v", voice, "--", input_line],
                    universal_newlines=True,
                )
            )

            if not ipa_str:
                _LOGGER.warning("No pronunciation for %s", word)
                continue

            ipa_pron = Pronunciation.from_string(ipa_str)

            yield word, ipa_pron.text, pos_tag


# -----------------------------------------------------------------------------


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    zip_args = [iter(iterable)] * n
    return itertools.zip_longest(*zip_args, fillvalue=fillvalue)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
