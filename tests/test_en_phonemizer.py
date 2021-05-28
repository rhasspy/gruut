#!/usr/bin/env python3
"""Tests for EnglishPhonemizer class"""
import unittest

from gruut import Token
from gruut.lang import get_phonemizer


class EnglishPhonemizerTestCase(unittest.TestCase):
    """Test cases for EnglishPhonemizer class"""

    def test_g2p(self):
        """Test grapheme to phoneme model for guessing pronunciations"""
        phonemizer = get_phonemizer("en-us")
        expected_phonemes = ["t", "ˈɛ", "s", "t"]

        guessed_prons = phonemizer.guess_pronunciations(Token("test"))
        self.assertGreater(len(guessed_prons), 0)

        actual_phonemes = guessed_prons[0].phonemes
        self.assertEqual(expected_phonemes, actual_phonemes)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
