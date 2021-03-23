#!/usr/bin/env python3
"""Converts a text lexicon to an sqlite3 database"""
import argparse
import sqlite3
import sys


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
        "--pos", action="store_true", help="Lexicon includes part of speech"
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    word_casing = lambda x: x

    if args.casing == "lower":
        word_casing = str.lower
    elif args.casing == "upper":
        word_casing = str.upper

    # -------------------------------------------------------------------------

    conn = sqlite3.connect(args.database)
    conn.execute("DROP TABLE IF EXISTS word_phonemes")
    conn.execute(
        "CREATE TABLE word_phonemes "
        + "(id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT, pron_order INTEGER, phonemes TEXT, pos TEXT);"
    )

    # word -> pron_order
    pron_orders = {}

    if args.lexicon == "-":
        lexicon_file = sys.stdin
    else:
        lexicon_file = open(args.lexicon, "r")

    with lexicon_file, conn:
        for line in lexicon_file:
            line = line.strip()
            if (not line) or line.startswith(";"):
                # Skip blank lines and comments
                continue

            if args.pos:
                # With part of speech
                word, pos, phonemes = line.split(maxsplit=2)
            else:
                # Without part of speech
                word, phonemes = line.split(maxsplit=1)
                pos = ""

            word = word_casing(word)
            pron_order = pron_orders.get(word, 0)

            conn.execute(
                "INSERT INTO word_phonemes (word, pron_order, phonemes, pos) VALUES (?, ?, ?, ?)",
                (word, pron_order, phonemes, pos),
            )

            pron_orders[word] = pron_order + 1


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
