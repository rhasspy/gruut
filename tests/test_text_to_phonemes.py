#!/usr/bin/env python3
"""Tests for text_to_phonemes API"""
import concurrent.futures
import functools
import unittest

from gruut import text_to_phonemes, Sentence


class TextToPhonemesTestCase(unittest.TestCase):
    """Test cases for text_to_phonemes API"""

    def test_flat_phonemes(self):
        """Test flat phonemes return format"""
        expected_phonemes = [
            "ˈaɪ",
            "w",
            "ˈaʊ",
            "n",
            "d",
            "ð",
            "ə",
            "w",
            "ˈu",
            "n",
            "d",
            "‖",
            "ˈaɪ",
            "ɹ",
            "ɪ",
            "f",
            "j",
            "ˈu",
            "z",
            "ð",
            "ə",
            "ɹ",
            "ˈɛ",
            "f",
            "j",
            "ˌu",
            "z",
            "‖",
        ]
        actual_phonemes = text_to_phonemes(
            "I wound the wound. I refuse the refuse.", return_format="flat_phonemes"
        )

        self.assertEqual(expected_phonemes, actual_phonemes)

    def test_word_phonemes(self):
        """Test word phonemes return format"""
        expected_phonemes = [
            ["ˈaɪ"],
            ["w", "ˈaʊ", "n", "d"],
            ["ð", "ə"],
            ["w", "ˈu", "n", "d"],
            ["‖"],
            ["ˈaɪ"],
            ["ɹ", "ɪ", "f", "j", "ˈu", "z"],
            ["ð", "ə"],
            ["ɹ", "ˈɛ", "f", "j", "ˌu", "z"],
            ["‖"],
        ]
        actual_phonemes = text_to_phonemes(
            "I wound the wound. I refuse the refuse.", return_format="word_phonemes"
        )

        self.assertEqual(expected_phonemes, actual_phonemes)

    def test_sentence_phonemes(self):
        """Test sentence phonemes return format"""
        expected_phonemes = [
            ["ˈaɪ", "w", "ˈaʊ", "n", "d", "ð", "ə", "w", "ˈu", "n", "d", "‖"],
            [
                "ˈaɪ",
                "ɹ",
                "ɪ",
                "f",
                "j",
                "ˈu",
                "z",
                "ð",
                "ə",
                "ɹ",
                "ˈɛ",
                "f",
                "j",
                "ˌu",
                "z",
                "‖",
            ],
        ]
        actual_phonemes = text_to_phonemes(
            "I wound the wound. I refuse the refuse.", return_format="sentence_phonemes"
        )

        self.assertEqual(expected_phonemes, actual_phonemes)

    def test_sentence_word_phonemes(self):
        """Test sentence and word phonemes return format"""
        expected_phonemes = [
            [["ˈaɪ"], ["w", "ˈaʊ", "n", "d"], ["ð", "ə"], ["w", "ˈu", "n", "d"], ["‖"]],
            [
                ["ˈaɪ"],
                ["ɹ", "ɪ", "f", "j", "ˈu", "z"],
                ["ð", "ə"],
                ["ɹ", "ˈɛ", "f", "j", "ˌu", "z"],
                ["‖"],
            ],
        ]
        actual_phonemes = text_to_phonemes(
            "I wound the wound. I refuse the refuse.",
            return_format="sentence_word_phonemes",
        )

        self.assertEqual(expected_phonemes, actual_phonemes)

    def test_word_tuples(self):
        """Test word tuples return format"""
        expected_tuples = [
            (0, "i", ["ˈaɪ"]),
            (0, "wound", ["w", "ˈaʊ", "n", "d"]),
            (0, "the", ["ð", "ə"]),
            (0, "wound", ["w", "ˈu", "n", "d"]),
            (0, ".", ["‖"]),
            (1, "i", ["ˈaɪ"]),
            (1, "refuse", ["ɹ", "ɪ", "f", "j", "ˈu", "z"]),
            (1, "the", ["ð", "ə"]),
            (1, "refuse", ["ɹ", "ˈɛ", "f", "j", "ˌu", "z"]),
            (1, ".", ["‖"]),
        ]
        actual_tuples = text_to_phonemes(
            "I wound the wound. I refuse the refuse.", return_format="word_tuples"
        )

        self.assertEqual(expected_tuples, actual_tuples)

    def test_sentences(self):
        """Test sentences return format"""
        expected_phonemes = [
            [["ˈaɪ"], ["w", "ˈaʊ", "n", "d"], ["ð", "ə"], ["w", "ˈu", "n", "d"], ["‖"]],
            [
                ["ˈaɪ"],
                ["ɹ", "ɪ", "f", "j", "ˈu", "z"],
                ["ð", "ə"],
                ["ɹ", "ˈɛ", "f", "j", "ˌu", "z"],
                ["‖"],
            ],
        ]
        actual_sentences = text_to_phonemes(
            "I wound the wound. I refuse the refuse.", return_format="sentences"
        )

        # Returns two Sentence objects
        self.assertEqual(len(actual_sentences), 2)
        self.assertIsInstance(actual_sentences[0], Sentence)
        self.assertIsInstance(actual_sentences[1], Sentence)

        self.assertEqual(expected_phonemes[0], actual_sentences[0].phonemes)
        self.assertEqual(expected_phonemes[1], actual_sentences[1].phonemes)

    def test_executor(self):
        """Test use in a ThreadPoolExecutor"""
        texts = ["All work and no play makes Jack a very dull person."] * 100

        with concurrent.futures.ThreadPoolExecutor() as executor:
            sentences = list(executor.map(functools.partial(text_to_phonemes), texts))

        self.assertEqual(len(sentences), len(texts))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
