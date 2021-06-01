#!/usr/bin/env python3
"""Tests for FrenchPhonemizer class"""
import unittest

from gruut import text_to_phonemes, Token
from gruut.lang import get_phonemizer, get_tokenizer


class FrenchPhonemizerTestCase(unittest.TestCase):
    """Test cases for FrenchPhonemizer class"""

    def test_liason(self):
        """Test addition of liason"""
        tokenizer = get_tokenizer("fr-fr")
        sentence = next(tokenizer.tokenize("J’ai des petites oreilles."))

        liason_word = "petites"
        self.assertIn(liason_word, sentence.clean_words)

        # Verify no liason
        phonemizer_no_liason = get_phonemizer("fr-fr", fr_no_liason=True)
        phonemes_no_liason = next(phonemizer_no_liason.phonemize(sentence.tokens))

        for word, word_phonemes in zip(sentence.clean_words, phonemes_no_liason):
            if word == liason_word:
                self.assertEqual(phonemes, ["p", "ə", "t", "i", "t"])

        # Verify liason
        phonemizer_liason = get_phonemizer("fr-fr")
        phonemes_liason = next(phonemizer_liason.phonemize(sentence.tokens))

        for word, word_phonemes in zip(sentence.clean_words, phonemes_no_liason):
            if word == liason_word:
                self.assertEqual(phonemes, ["p", "ə", "t", "i", "t", "z"])

    def test_last_token(self):
        """Ensure liason does not leave last token"""
        phonemes = text_to_phonemes("Est-ce-que", lang="fr")
        self.assertGreater(len(phonemes), 0)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
