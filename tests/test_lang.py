#!/usr/bin/env python3
"""Tests for gruut.lang"""
import unittest

from gruut.lang import id_to_phonemes


class GruutLangTestCase(unittest.TestCase):
    """Test cases for gruut.lang"""

    def test_id_to_phonemes(self):
        """Test id_to_phonemes"""
        phonemes = id_to_phonemes("en-us")
        self.assertEqual(
            phonemes,
            [
                "_",  # pad
                "|",  # minor break
                "‖",  # major break
                "#",  # word break
                "ˈ",  # primary stress
                "ˌ",  # secondary stress
                "aɪ",
                "aʊ",
                "b",
                "d",
                "d͡ʒ",
                "eɪ",
                "f",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "oʊ",
                "p",
                "s",
                "t",
                "t͡ʃ",
                "u",
                "v",
                "w",
                "z",
                "æ",
                "ð",
                "ŋ",
                "ɑ",
                "ɔ",
                "ɔɪ",
                "ə",
                "ɚ",
                "ɛ",
                "ɡ",
                "ɪ",
                "ɹ",
                "ʃ",
                "ʊ",
                "ʌ",
                "ʒ",
                "θ",
            ]
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
