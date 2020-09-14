#!/usr/bin/env python3
"""Tests for Phonemes class"""
import unittest
from pathlib import Path

from gruut import Phonemes

_DIR = Path(__file__).parent


class PhonemesTestCase(unittest.TestCase):
    """Test cases for Phonemes class"""

    def test_from_string(self):
        """Test Phonemes.from_string"""
        # "Just a cow."
        pron_str = "/dʒʌst ə kaʊ/"

        phonemes_path = _DIR.parent / "gruut" / "data" / "en-us" / "phonemes.txt"
        with open(phonemes_path, "r") as phonemes_file:
            phonemes = Phonemes.from_text(phonemes_file)

        pron_phonemes = phonemes.split(pron_str)

        # Ensure "d ʒ" -> "d͡ʒ" and "a ʊ" -> "aʊ"
        phoneme_strs = [p.text for p in pron_phonemes]
        self.assertEqual(phoneme_strs, ["d͡ʒ", "ʌ", "s", "t", "ə", "k", "aʊ"])


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
