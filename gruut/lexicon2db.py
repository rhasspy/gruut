#!/usr/bin/env python3
"""Converts a text lexicon to a gruut sqlite3 database"""
import argparse
import sys

from .const import TokenFeatures, WordPronunciation
from .phonemize import SqlitePhonemizer

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="lexicon2db.py")
    parser.add_argument(
        "--casing",
        required=True,
        choices=("keep", "lower", "upper"),
        help="Casing to apply to words",
    )
    parser.add_argument(
        "--lexicon",
        required=True,
        help="Text lexicon to read with <WORD> <PHONEME> <PHONEME> ...",
    )
    parser.add_argument("--database", required=True, help="SQLite database to write")
    parser.add_argument(
        "--pos",
        action="store_true",
        help="Lexicon includes part of speech (2nd column)",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    word_casing = None

    if args.casing == "lower":
        word_casing = str.lower
    elif args.casing == "upper":
        word_casing = str.upper

    # -------------------------------------------------------------------------

    token_features = []
    if args.pos:
        token_features.append(TokenFeatures.PART_OF_SPEECH)

    phonemizer = SqlitePhonemizer(
        database_path=args.database, token_features=token_features
    )
    phonemizer.create_tables(drop_existing=True)

    # word -> pron_order
    pron_orders = {}

    if args.lexicon == "-":
        lexicon_file = sys.stdin
    else:
        lexicon_file = open(args.lexicon, "r")

    with lexicon_file, phonemizer.db_conn:
        for i, line in enumerate(lexicon_file):
            try:
                line = line.strip()
                if (not line) or line.startswith(";"):
                    # Skip blank lines and comments
                    continue

                if args.pos:
                    # With part of speech
                    word, pos_str, phonemes_str = line.split(maxsplit=2)
                    pos = set(pos_str.split(","))
                    phonemes = phonemes_str.split()

                    word_pron = WordPronunciation(
                        phonemes, preferred_features={TokenFeatures.PART_OF_SPEECH: pos}
                    )
                else:
                    # Without part of speech
                    word, phonemes_str = line.split(maxsplit=1)
                    phonemes = phonemes_str.split()
                    word_pron = WordPronunciation(phonemes)

                if word_casing:
                    word = word_casing(word)

                pron_order = pron_orders.get(word, 0)

                # Don't commit on every word, or it will be terribly slow
                phonemizer.insert_prons(word, [word_pron], commit=False)

                pron_orders[word] = pron_order + 1
            except Exception as e:
                print("Error on line", i + 1, "-", line)
                raise e


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
