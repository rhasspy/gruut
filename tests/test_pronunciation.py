#!/usr/bin/env python3
"""Tests for Pronunciation class"""
import unittest

from gruut import Pronunciation


class PronunciationTestCase(unittest.TestCase):
    """Test cases for Pronunciation class"""

    def test_from_string(self):
        """Test Pronuncation.from_string"""
        # "Yes, choose IPA."
        pron_str = "ˈjɛs|ˈt͡ʃuːz aɪpiːeɪ‖"

        pron = Pronunciation.from_string(pron_str, keep_stress=False)

        phone_strs = [p.text for p in pron.phones]
        self.assertEqual(
            phone_strs, ["j", "ɛ", "s", "t͡ʃ", "uː", "z", "a", "ɪ", "p", "iː", "e", "ɪ"]
        )

        phone_break_strs = [pb.text for pb in pron.phones_and_breaks]
        self.assertEqual(
            phone_break_strs,
            ["j", "ɛ", "s", "|", "t͡ʃ", "uː", "z", "a", "ɪ", "p", "iː", "e", "ɪ", "‖"],
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
