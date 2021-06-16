#!/usr/bin/env python3
"""Command-line interface to gruut"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import jsonlines

from gruut_ipa import IPA

from .utils import find_lang_dir

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut")

# Path to gruut base directory
_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        # Print known languages and exit
        from .lang import KNOWN_LANGS

        print("Languages:", *sorted(list(KNOWN_LANGS)))
        sys.exit(0)
    elif sys.argv[1] == "--version":
        from . import __version__

        print(__version__)
        sys.exit(0)

    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    if args.lang_dir:
        args.lang_dir = Path(args.lang_dir)
    else:
        args.lang_dir = find_lang_dir(args.language)

    args.func(args)


# -----------------------------------------------------------------------------


def do_tokenize(args):
    """
    Split lines from stdin into sentences, tokenize and clean.

    Prints a line of JSON for each sentence.
    """
    from .commands import tokenize
    from .lang import get_tokenizer

    tokenizer = get_tokenizer(
        args.language,
        lang_dir=args.lang_dir,
        no_pos=args.no_pos,
        use_number_converters=args.number_converters,
        do_replace_currency=(not args.disable_currency),
        exclude_non_words=(not args.no_exclude_non_words),
    )

    if args.text:
        # Use arguments
        lines = args.text
    else:
        # Use stdin
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for sent_json in tokenize(
        tokenizer,
        lines,
        is_csv=args.csv,
        csv_delimiter=args.csv_delimiter,
        split_sentences=args.split_sentences,
    ):
        writer.write(sent_json)


# -----------------------------------------------------------------------------


def do_phonemize(args):
    """
    Reads JSONL from stdin with "clean_words" property.

    Looks up or guesses phonetic pronuncation(s) for all clean words.

    Prints a line of JSON for each input line.
    """
    from .lang import get_phonemizer
    from .commands import phonemize

    word_break = IPA.BREAK_WORD if args.word_breaks else None
    phonemizer = get_phonemizer(
        args.language,
        args.lang_dir,
        use_word_indexes=args.word_indexes,
        word_break=word_break,
        no_g2p=args.no_g2p,
        model_prefix=args.model_prefix,
    )

    if os.isatty(sys.stdin.fileno()):
        print("Reading tokenize JSONL from stdin...", file=sys.stderr)

    def sentence_generator():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            yield json.loads(line)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for utt_json in phonemize(phonemizer, sentence_generator()):
        writer.write(utt_json)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="gruut")
    parser.add_argument("language", help="Language code (e.g., en-us)")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # --------
    # tokenize
    # --------
    tokenize_parser = sub_parsers.add_parser(
        "tokenize", help="Sentencize/tokenize raw text, clean, and expand numbers"
    )
    tokenize_parser.add_argument(
        "text", nargs="*", help="Text to tokenize (default: stdin)"
    )
    tokenize_parser.add_argument(
        "--disable-currency",
        action="store_true",
        help="Disable automatic replacement of currency with words (e.g., $1 -> one dollar)",
    )
    tokenize_parser.add_argument(
        "--number-converters",
        action="store_true",
        help="Allow number_conv form for specifying num2words converter (cardinal, ordinal, ordinal_num, year, currency)",
    )
    tokenize_parser.add_argument(
        "--no-exclude-non-words",
        action="store_true",
        help="Don't remove punctionation, etc. from clean words",
    )
    tokenize_parser.add_argument(
        "--split-sentences",
        action="store_true",
        help="Output one line for every sentence",
    )
    tokenize_parser.add_argument(
        "--no-pos",
        action="store_true",
        help="Don't load part of speech tagger if available",
    )
    tokenize_parser.add_argument(
        "--csv", action="store_true", help="Input format is id|text"
    )
    tokenize_parser.add_argument(
        "--csv-delimiter",
        default="|",
        help="Delimiter between id and text (default: |, requires --csv)",
    )
    tokenize_parser.set_defaults(func=do_tokenize)

    # ---------
    # phonemize
    # ---------
    phonemize_parser = sub_parsers.add_parser(
        "phonemize", help="Look up or guess word pronunciations from JSONL sentences"
    )
    phonemize_parser.set_defaults(func=do_phonemize)
    phonemize_parser.add_argument(
        "--word-separator",
        default=" ",
        help="Separator to add between words in output pronunciation (default: space)",
    )
    phonemize_parser.add_argument(
        "--phoneme-separator",
        default=" ",
        help="Separator to add between words in output pronunciation (default: space)",
    )
    phonemize_parser.add_argument(
        "--word-indexes",
        action="store_true",
        help="Allow word_n form for specifying nth pronunciation of word from lexicon",
    )
    phonemize_parser.add_argument(
        "--word-breaks",
        action="store_true",
        help="Add the IPA word break symbol (#) between each word",
    )
    phonemize_parser.add_argument(
        "--no-g2p",
        action="store_true",
        help="Don't load grapheme to phoneme model if available for guessing pronunciations",
    )
    phonemize_parser.add_argument(
        "--fail-on-unknown-words",
        action="store_true",
        help="Raise an error if there are words whose pronunciations can't be guessed",
    )
    phonemize_parser.add_argument(
        "--model-prefix",
        help="Directory to use within default language directory with different lexicon/g2p model (e.g., espeak)",
    )
    # phonemize_parser.add_argument(
    #     "--skip-on-unknown-words",
    #     action="store_true",
    #     help="Skip sentences with words whose pronunciations can't be guessed",
    # )

    # ----------------
    # Shared arguments
    # ----------------
    for sub_parser in [tokenize_parser, phonemize_parser]:
        sub_parser.add_argument(
            "--lang-dir", help="Directory with language-specific data files"
        )
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
