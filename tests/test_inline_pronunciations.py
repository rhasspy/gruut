#!/usr/bin/env python3
"""Tests for inline pronunciations"""
import unittest

import gruut_ipa
from gruut import text_to_phonemes
from gruut.utils import encode_inline_pronunciations, decode_inline_pronunciation


class InlinePronunciationsTestCase(unittest.TestCase):
    """Test cases for inline pronunciations"""

    def test_encode_decode(self):
        """Test encoding/decoding of inline pronunciations"""
        text = "This is a [[ t ɛ s t ]]"

        phonemes = gruut_ipa.Phonemes.from_language("en-us")
        encoded_words = encode_inline_pronunciations(text, phonemes).split()
        decoded_word = decode_inline_pronunciation(encoded_words[-1])

        self.assertEqual(decoded_word, "t ɛ s t")

    def test_text_to_phonemes(self):
        """Test inline pronunciations in text_to_phonemes"""
        text = "This is a [[ t ɛ s t ]]"
        words_phonemes = list(text_to_phonemes(text, inline_pronunciations=True))
        _, last_word, last_phonemes = words_phonemes[-1]

        self.assertEqual(last_word, "[[ t ɛ s t ]]")
        self.assertEqual(last_phonemes, ["t", "ɛ", "s", "t"])


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
