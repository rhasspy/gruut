#!/usr/bin/env python3
"""Converts a text lexicon to a gruut sqlite3 database"""
import argparse
import sqlite3
import sys

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
        "--role", action="store_true", help="Lexicon includes word roles (2nd column)",
    )
    parser.add_argument(
        "--empty-role",
        default="_",
        help="String used to identify empty word role (see --role)",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    word_casing = None

    if args.casing == "lower":
        word_casing = str.lower
    elif args.casing == "upper":
        word_casing = str.upper

    # -------------------------------------------------------------------------

    conn = sqlite3.connect(args.database)

    # Re-create tables in output
    conn.execute("DROP TABLE IF EXISTS word_phonemes")
    conn.execute("DROP TABLE IF EXISTS g2p_alignments")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS word_phonemes "
        + "(id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT, pron_order INTEGER, phonemes TEXT, role TEXT);"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS g2p_alignments "
        + "(id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT, alignment TEXT);"
    )
    conn.commit()

    # word -> pron_order
    pron_orders = {}

    if args.lexicon == "-":
        lexicon_file = sys.stdin
    else:
        lexicon_file = open(args.lexicon, "r", encoding="utf-8")

    with lexicon_file, conn:
        for i, line in enumerate(lexicon_file):
            try:
                line = line.strip()
                if (not line) or line.startswith(";") or (" " not in line):
                    # Skip blank lines and comments.
                    # Also skip lines without a pronunciation.
                    continue

                role = ""

                if args.role:
                    # With role
                    word, role, phonemes_str = line.split(maxsplit=2)

                    if role == args.empty_role:
                        role = ""
                    elif ":" not in role:
                        role = f"gruut:{role}"
                else:
                    # Without part of speech
                    word, phonemes_str = line.split(maxsplit=1)

                if word_casing:
                    word = word_casing(word)

                pron_order = pron_orders.get(word, 0)

                # Don't commit on every word, or it will be terribly slow
                conn.execute(
                    "INSERT into word_phonemes (word, pron_order, phonemes, role) VALUES (?, ?, ?, ?)",
                    (word, pron_order, phonemes_str, role),
                )

                pron_orders[word] = pron_order + 1
            except Exception as e:
                print("Error on line", i + 1, "-", line)
                raise e


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
