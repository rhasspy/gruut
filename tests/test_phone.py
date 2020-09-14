#!/usr/bin/env python3
"""Tests for Phone class"""
import unittest

from gruut import IPA, Phone, Stress, VowelHeight, VowelPlacement


class PhoneTestCase(unittest.TestCase):
    """Test cases for Phone class"""

    def test_from_string(self):
        """Test Phone.from_string"""
        # ˈãː
        codepoints = [IPA.STRESS_PRIMARY, "a", IPA.NASAL, IPA.LONG]
        ipa = "".join(codepoints)

        phone = Phone.from_string(ipa)
        self.assertEqual(phone.text, "ˈãː")
        self.assertEqual(phone.letters, "a")
        self.assertEqual(phone.diacritics, {IPA.NASAL})
        self.assertEqual(phone.suprasegmentals, {IPA.STRESS_PRIMARY, IPA.LONG})

        self.assertEqual(phone.stress, Stress.PRIMARY)
        self.assertEqual(phone.is_nasal, True)
        self.assertEqual(phone.is_long, True)

        self.assertEqual(phone.is_vowel, True)
        self.assertEqual(phone.vowel.height, VowelHeight.OPEN)
        self.assertEqual(phone.vowel.placement, VowelPlacement.FRONT)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
