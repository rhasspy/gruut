#!/usr/bin/env python3
import sqlite3
import sys

import gruut_ipa

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    conn = sqlite3.connect(sys.argv[1])
    cursor = conn.execute("SELECT phonemes FROM word_phonemes")

    phonemes = set()
    for row in cursor:
        phoneme_str = row[0]
        for phoneme in phoneme_str.split():
            phoneme = gruut_ipa.IPA.without_stress(phoneme)
            phonemes.add(phoneme)

    for phoneme in sorted(phonemes):
        print(phoneme)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
