#!/usr/bin/env python3
"""Command-line interface to gruut"""
import argparse
import dataclasses
import logging
import os
import sys
from pathlib import Path

import jsonlines

from gruut import get_text_processor
from gruut.const import KNOWN_LANGS
from gruut.utils import print_graph

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut")

# Path to gruut base directory
_DIR = Path(__file__).parent

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

    if args.lang_dir:
        args.lang_dir = Path(args.lang_dir)

    # -------------------------------------------------------------------------

    text_processor = get_text_processor(
        default_lang=args.language,
        languages=[args.language] if not args.ssml else None,
        load_pos_tagger=(not args.no_pos),
        load_phoneme_lexicon=(not args.no_lexicon),
        load_g2p_guesser=(not args.no_g2p),
        model_prefix=args.model_prefix,
        lang_dir=args.lang_dir,
    )

    if args.debug:
        _LOGGER.debug(text_processor.settings)

    if args.text:
        # Use arguments
        lines = args.text
    else:
        # Use stdin
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in lines:
        if not line.strip():
            continue

        graph, root = text_processor(
            line,
            ssml=args.ssml,
            pos=(not args.no_pos),
            phonemize=(not (args.no_lexicon and args.no_g2p)),
            post_process=(not args.no_post_process),
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
        for sentence in text_processor.sentences(
            graph,
            root,
            major_breaks=(not args.no_major_breaks),
            minor_breaks=(not args.no_minor_breaks),
            punctuations=(not args.no_punctuation),
        ):
            sentence_dict = dataclasses.asdict(sentence)
            writer.write(sentence_dict)


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

    # Miscellaneous
    parser.add_argument(
        "--model-prefix",
        help="Sub-directory of gruut language data files with different lexicon, etc. (e.g., espeak)",
    )
    parser.add_argument(
        "--lang-dir", help="Directory with language-specific data files"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
