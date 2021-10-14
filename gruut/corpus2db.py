#!/usr/bin/env python3
"""Converts a Phonetisaurus G2P corpus to an sqlite database"""
import argparse
import sqlite3

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="corpus2db.py")
    parser.add_argument(
        "--corpus", required=True, help="Phonetisaurus G2P corpus file to read"
    )
    parser.add_argument("--database", required=True, help="SQLite database to write")
    parser.add_argument(
        "--no-drop",
        action="store_true",
        help="Don't drop existing g2p_alignments table",
    )
    args = parser.parse_args()

    # -------------------------------------------------------------------------

    conn = sqlite3.connect(args.database)

    g2p_alignments = {}
    with open(args.corpus, "r", encoding="utf-8") as corpus_file:
        for line in corpus_file:
            line = line.strip()
            if not line:
                continue

            word = ""

            # Parse line
            parts = line.split()
            for part in parts:
                # Assume default delimiters:
                # } separates input/output
                # | separates input/output tokens
                # _ indicates empty output
                part_in, _part_out = part.split("}")
                part_ins = part_in.split("|")
                word += "".join(part_ins)

            if word and (word not in g2p_alignments):
                g2p_alignments[word] = line

    # Add to database
    with conn:
        if not args.no_drop:
            conn.execute("DROP TABLE IF EXISTS g2p_alignments")

        conn.execute(
            "CREATE TABLE IF NOT EXISTS g2p_alignments "
            + "(id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT, alignment TEXT);"
        )

        for word, alignment in g2p_alignments.items():
            conn.execute(
                "INSERT INTO g2p_alignments (word, alignment) VALUES (?, ?)",
                (word, alignment),
            )

    print("Added", len(g2p_alignments), "alignments to", args.database)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
