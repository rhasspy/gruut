#!/usr/bin/env python3
"""Tests for text_to_phonemes API"""
import unittest

from gruut import text_to_phonemes, Sentence


class TextToPhonemesTestCase(unittest.TestCase):
    """Test cases for text_to_phonemes API"""

    def test_en(self):
        """Test English sentence with ambiguous pronunciations"""
        expected_phonemes = [
            [["ˈaɪ"], ["w", "ˈaʊ", "n", "d"], ["ð", "ə"], ["w", "ˈu", "n", "d"], ["‖"]]
        ]
        actual_phonemes = text_to_phonemes("I wound the wound.")

        self.assertEqual(expected_phonemes, actual_phonemes)

    def test_en_sentences(self):
        """Test return_sentences=True"""
        expected_phonemes = [
            ["ˈaɪ"],
            ["w", "ˈaʊ", "n", "d"],
            ["ð", "ə"],
            ["w", "ˈu", "n", "d"],
            ["‖"],
        ]
        actual_sentences = text_to_phonemes("I wound the wound.", return_sentences=True)
        self.assertEqual(len(actual_sentences), 1)
        self.assertIsInstance(actual_sentences[0], Sentence)

        self.assertEqual(expected_phonemes, actual_sentences[0].phonemes)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
