#!/usr/bin/env python3
"""Tests for inline pronunciations"""
import unittest

import gruut_ipa
from gruut import text_to_phonemes
from gruut.utils import (
    encode_inline_pronunciations,
    decode_inline_pronunciation,
    InlinePronunciationType,
)


class InlinePronunciationsTestCase(unittest.TestCase):
    """Test cases for inline pronunciations"""

    def test_encode_decode(self):
        """Test encoding/decoding of inline pronunciations"""
        text = "This is a [[ t ɛ s t ]]"

        phonemes = gruut_ipa.Phonemes.from_language("en-us")
        encoded_words = encode_inline_pronunciations(text, phonemes).split()
        inline_type, decoded_word = decode_inline_pronunciation(encoded_words[-1])

        self.assertEqual(inline_type, InlinePronunciationType.PHONEMES)
        self.assertEqual(decoded_word, "t ɛ s t")

    def test_encode_decode_sounds_like(self):
        """Test encoding/decoding of inline sounds-like pronunciations"""
        text = "This is {{ a test }}"

        phonemes = gruut_ipa.Phonemes.from_language("en-us")
        encoded_words = encode_inline_pronunciations(text, phonemes).split()
        inline_type, decoded_word = decode_inline_pronunciation(encoded_words[-1])

        self.assertEqual(inline_type, InlinePronunciationType.SOUNDS_LIKE)
        self.assertEqual(decoded_word, "a test")

    def test_text_to_phonemes(self):
        """Test inline pronunciations in text_to_phonemes"""
        text = "This is a [[ t ɛ s t ]]"
        words_phonemes = list(text_to_phonemes(text, inline_pronunciations=True))
        _, last_word, last_phonemes = words_phonemes[-1]

        self.assertEqual(last_word, "[[ t ɛ s t ]]")
        self.assertEqual(last_phonemes, ["t", "ɛ", "s", "t"])

    def test_text_to_phonemes_sounds_like(self):
        """Test inline sounds-like pronunciations in text_to_phonemes"""
        text = "{{ racks uh core {i}t {co}de {fall}{i}ble {pu}n tore s{ee} us }}"
        words_phonemes = list(text_to_phonemes(text, inline_pronunciations=True))
        _, last_word, last_phonemes = words_phonemes[-1]

        self.assertEqual(last_word, text)
        self.assertEqual(
            last_phonemes,
            [
                "ɹ",
                "ˈæ",
                "k",
                "s",
                "ˈʌ",
                "k",
                "ˈɔ",
                "ɹ",
                "ˈɪ",
                "k",
                "ˈoʊ",
                "f",
                "ˈæ",
                "l",
                "ə",
                "p",
                "ˈʌ",
                "t",
                "ˈɔ",
                "ɹ",
                "ˈi",
                "ˈʌ",
                "s",
            ],
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
