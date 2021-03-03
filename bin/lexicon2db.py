#!/usr/bin/env python3
"""Converts a text lexicon to an sqlite3 database"""
import argparse
import sqlite3


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="lexicon2db.py")
    parser.add_argument(
        "lexicon", help="Text lexicon to read with <WORD> <PHONEME> <PHONEME> ..."
    )
    parser.add_argument("database", help="SQLite database to write")
    args = parser.parse_args()

    conn = sqlite3.connect(args.database)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS word_phonemes "
        + "(id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT, phonemes TEXT);"
    )

    with open(args.lexicon, "r") as lexicon_file, conn:
        for line in lexicon_file:
            line = line.strip()
            if (not line) or line.startswith(";"):
                # Skip blank lines and comments
                continue

            word, phonemes = line.split(maxsplit=1)
            conn.execute(
                "INSERT INTO word_phonemes (word, phonemes) VALUES (?, ?)",
                (word, phonemes),
            )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
