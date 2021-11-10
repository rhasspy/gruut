#!/usr/bin/env python3
"""Command-line interface to gruut"""
import argparse
import csv
import dataclasses
import logging
import os
import sys
from enum import Enum
from pathlib import Path

import jsonlines

from gruut.const import KNOWN_LANGS
from gruut.text_processor import TextProcessor
from gruut.utils import print_graph

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut")

# Path to gruut base directory
_DIR = Path(__file__).parent


class StdinFormat(str, Enum):
    """Format of standard input"""

    AUTO = "auto"
    """Choose based on SSML state"""

    LINES = "lines"
    """Each line is a separate sentence/document"""

    DOCUMENT = "document"
    """Entire input is one document"""


# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        # Print known languages and exit
        print("Languages:", *sorted(list(KNOWN_LANGS)))
        sys.exit(0)
    elif sys.argv[1] == "--version":
        from gruut import __version__

        print(__version__)
        sys.exit(0)

    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    if args.espeak:
        args.model_prefix = "espeak"

    # -------------------------------------------------------------------------

    text_processor = TextProcessor(
        default_lang=args.language, model_prefix=args.model_prefix,
    )

    if args.debug:
        _LOGGER.debug(text_processor.settings)

    if args.text:
        # Use arguments
        lines = args.text
    else:
        # Use stdin
        stdin_format = StdinFormat.LINES

        if (args.stdin_format == StdinFormat.AUTO) and args.ssml:
            # Assume SSML input is entire document
            stdin_format = StdinFormat.DOCUMENT

        if stdin_format == StdinFormat.DOCUMENT:
            # One big line
            lines = [sys.stdin.read()]
        else:
            # Multiple lines
            lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading input from stdin...", file=sys.stderr)

    if args.csv:
        writer = csv.writer(sys.stdout, delimiter=args.csv_delimiter)

        def input_text(lines):
            reader = csv.reader(lines, delimiter=args.csv_delimiter)
            for row in reader:
                text = row[1]
                yield (text, row)

        def output_sentences(sentences, writer, text_data=None):
            row = list(text_data)
            row.append(
                args.sentence_separator.join(
                    args.word_separator.join(w.text for w in sentence if w.is_spoken)
                    for sentence in sentences
                )
            )

            phonemes = [
                args.phoneme_separator.join(w.phonemes)
                for sentence in sentences
                for w in sentence
                if w.phonemes
            ]

            row.append(args.phoneme_word_separator.join(phonemes))
            writer.writerow(row)

    else:
        writer = jsonlines.Writer(sys.stdout, flush=True)

        def input_text(lines):
            for line in lines:
                yield (line, None)

        def output_sentences(sentences, writer, text_data=None):
            for sentence in sentences:
                sentence_dict = dataclasses.asdict(sentence)
                writer.write(sentence_dict)

    for text, text_data in input_text(lines):
        try:
            graph, root = text_processor(
                text,
                ssml=args.ssml,
                pos=(not args.no_pos),
                phonemize=(not (args.no_lexicon and args.no_g2p)),
                post_process=(not args.no_post_process),
                verbalize_numbers=(not args.no_numbers),
                verbalize_currency=(not args.no_currency),
                verbalize_dates=(not args.no_dates),
                verbalize_times=(not args.no_times),
            )

            if args.debug:
                print_graph(
                    graph,
                    root,
                    print_func=lambda *print_args: _LOGGER.debug(
                        " ".join(str(a) for a in print_args)
                    ),
                )

            # Output sentences
            sentences = list(
                text_processor.sentences(
                    graph,
                    root,
                    major_breaks=(not args.no_major_breaks),
                    minor_breaks=(not args.no_minor_breaks),
                    punctuations=(not args.no_punctuation),
                )
            )

            output_sentences(sentences, writer, text_data)
        except Exception as e:
            _LOGGER.exception(text)

            if not args.no_fail:
                raise TextProcessingError(text) from e


# -----------------------------------------------------------------------------


class TextProcessingError(Exception):
    """Raised when a line of input results in an exception"""

    pass


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="gruut")
    parser.add_argument(
        "-l",
        "--language",
        default="en-us",
        help="Default language code (default: en-us)",
    )

    parser.add_argument("text", nargs="*", help="Text to tokenize (default: stdin)")
    parser.add_argument(
        "--ssml", action="store_true", help="Input text is SSML",
    )
    parser.add_argument(
        "--stdin-format",
        choices=[str(v.value) for v in StdinFormat],
        default=StdinFormat.AUTO,
        help="Format of stdin text (default: auto)",
    )

    # Disable features
    parser.add_argument(
        "--no-numbers",
        action="store_true",
        help="Disable number replacement (1 -> one)",
    )
    parser.add_argument(
        "--no-currency",
        action="store_true",
        help="Disable currency replacement ($1 -> one dollar)",
    )
    parser.add_argument(
        "--no-dates",
        action="store_true",
        help="Disable date replacement (4/1/2021 -> April first twenty twenty one)",
    )
    parser.add_argument(
        "--no-times",
        action="store_true",
        help="Disable time replacement (4:01pm -> four oh one P M)",
    )
    parser.add_argument(
        "--no-pos", action="store_true", help="Disable part of speech tagger",
    )
    parser.add_argument(
        "--no-lexicon", action="store_true", help="Disable phoneme lexicon database",
    )
    parser.add_argument(
        "--no-g2p", action="store_true", help="Disable grapheme to phoneme guesser",
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="Don't output punctuations (quotes, brackets, etc.)",
    )
    parser.add_argument(
        "--no-major-breaks",
        action="store_true",
        help="Don't output major breaks (periods, question marks, etc.)",
    )
    parser.add_argument(
        "--no-minor-breaks",
        action="store_true",
        help="Don't output minor breaks (commas, semicolons, etc.)",
    )
    parser.add_argument(
        "--no-post-process",
        action="store_true",
        help="Disable post-processing of sentences (e.g., liasons)",
    )
    parser.add_argument(
        "--no-fail", action="store_true", help="Skip lines that result in errors",
    )

    # Miscellaneous
    parser.add_argument(
        "--espeak",
        action="store_true",
        help="Use eSpeak versions of lexicons (overrides --model-prefix)",
    )
    parser.add_argument(
        "--model-prefix",
        help="Sub-directory of gruut language data files with different lexicon, etc. (e.g., espeak)",
    )
    parser.add_argument(
        "--csv", action="store_true", help="Input text is id|text (see --csv-delimiter)"
    )
    parser.add_argument(
        "--csv-delimiter", default="|", help="Delimiter for input text with --csv"
    )
    parser.add_argument(
        "--sentence-separator",
        default=". ",
        help="String used to separate sentences in CSV output",
    )
    parser.add_argument(
        "--word-separator",
        default=" ",
        help="String used to separate words in CSV output",
    )
    parser.add_argument(
        "--phoneme-word-separator",
        default="#",
        help="String used to separate phonemes in CSV output",
    )
    parser.add_argument(
        "--phoneme-separator",
        default=" ",
        help="String used to separate words in CSV output phonemes",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
