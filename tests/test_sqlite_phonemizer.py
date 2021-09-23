#!/usr/bin/env python3
"""Tests for phonemization"""
import unittest

from gruut import sentences

# Translation from https://omniglot.com for:
# My hovercraft is full of eels.


class PhonemizerTestCase(unittest.TestCase):
    """Test cases for phonemization"""

    def test_ar(self):
        """Arabic test"""
        self.assertEqual(
            self.get_phonemes("حَوّامتي مُمْتِلئة", "ar"),
            [
                ("حَوّامتي", ["ħ", "aw", "w", "i", "m", "m", "t", "iː"]),
                ("مُمْتِلئة", ["m", "u", "m", "t", "i", "l", "l"],),
            ],
        )

    def test_cs(self):
        """Czech test"""
        self.assertEqual(
            self.get_phonemes("Moje vznášedlo je plné úhořů.", "cs-cz"),
            [
                ("Moje", ["m", "o", "j", "ɛ"]),
                ("vznášedlo", ["v", "z", "n", "aː", "ʃ", "ɛ", "d", "l", "o"]),
                ("je", ["j", "ɛ"]),
                ("plné", ["p", "l", "n", "ɛː"]),
                ("úhořů", ["uː", "ɦ", "o", "r̝", "uː"]),
                (".", ["‖"]),
            ],
        )

    def test_de_us(self):
        """German test"""
        self.assertEqual(
            self.get_phonemes("Mein Luftkissenfahrzeug ist voller Aale.", "de_DE"),
            [
                ("Mein", ["m", "aɪ̯", "n"]),
                (
                    "Luftkissenfahrzeug",
                    [
                        "l",
                        "ʊ",
                        "f",
                        "t",
                        "k",
                        "ɪ",
                        "s",
                        "z",
                        "ɛ",
                        "n",
                        "f",
                        "a",
                        "ʁ",
                        "t",
                        "s",
                        "ɔ",
                        "ɔʏ̯",
                        "k",
                    ],
                ),
                ("ist", ["ɪ", "s", "t"]),
                ("voller", ["v", "ɔ", "l", "l", "ɐ"]),
                ("Aale", ["ʔ", "aː", "l", "ə"]),
                (".", ["‖"]),
            ],
        )

    def test_en_us(self):
        """English test"""
        self.assertEqual(
            self.get_phonemes("My hovercraft is full of eels.", "en_US"),
            [
                ("My", ["m", "ˈaɪ"]),
                ("hovercraft", ["h", "ˈʌ", "v", "ɚ", "k", "ɹ", "ˌæ", "f", "t"],),
                ("is", ["ˈɪ", "z"]),
                ("full", ["f", "ˈʊ", "l"]),
                ("of", ["ə", "v"]),
                ("eels", ["ˈi", "l", "z"]),
                (".", ["‖"]),
            ],
        )

    def test_es(self):
        """Spanish test"""
        self.assertEqual(
            self.get_phonemes("Mi aerodeslizador está lleno de anguilas.", "es_ES"),
            [
                ("Mi", ["m", "i"]),
                (
                    "aerodeslizador",
                    [
                        "a",
                        "e",
                        "ɾ",
                        "o",
                        "d",
                        "e",
                        "s",
                        "l",
                        "i",
                        "θ",
                        "a",
                        "d",
                        "o",
                        "ɾ",
                    ],
                ),
                ("está", ["e", "s", "t", "a"]),
                ("lleno", ["ʎ", "e", "n", "o"]),
                ("de", ["d", "e"]),
                ("anguilas", ["a", "n", "g", "i", "l", "a", "s"]),
                (".", ["‖"]),
            ],
        )

    def test_fr(self):
        """French test"""
        self.assertEqual(
            self.get_phonemes("Mon aéroglisseur est plein d'anguilles.", "fr_FR"),
            [
                ("Mon", ["m", "ɔ̃", "n"]),
                ("aéroglisseur", ["a", "e", "ʁ", "ɔ", "ɡ", "l", "i", "s", "œ", "ʁ"],),
                ("est", ["ɛ"]),
                ("plein", ["p", "l", "ɛ̃"]),
                ("d'anguilles", ["d", "ɑ̃", "ɡ", "i", "j"]),
                (".", ["‖"]),
            ],
        )

    def test_it(self):
        """Italian test"""
        self.assertEqual(
            self.get_phonemes("Il mio hovercraft è pieno di anguille.", "it_IT"),
            [
                ("Il", ["i", "l"]),
                ("mio", ["ˈm", "i", "o"],),
                ("hovercraft", ["o", "v", "e", "r", "k", "r", "a", "f", "t"]),
                ("è", ["ɛ"]),
                ("pieno", ["ˈp", "j", "ɛ", "n", "o"]),
                ("di", ["ˈd", "i"]),
                ("anguille", ["a", "n", "ɡ", "w", "i", "l", "l", "e"]),
                (".", ["‖"]),
            ],
        )

    def test_nl(self):
        """Dutch test"""
        self.assertEqual(
            self.get_phonemes("Mijn luchtkussenboot zit vol paling.", "nl"),
            [
                ("Mijn", ["m", "ɛi", "n"]),
                (
                    "luchtkussenboot",
                    ["ˈl", "ʏ", "x", "t", "k", "ʏ", "s", "ə", "n", "ˌb", "o", "t"],
                ),
                ("zit", ["z", "ɪ", "t"]),
                ("vol", ["v", "ɔ", "l"]),
                ("paling", ["p", "a", "l", "ɪ", "ŋ"]),
                (".", ["‖"]),
            ],
        )

    def get_phonemes(self, text, lang):
        """Return (text, phonemes) for each word"""
        sentence = next(sentences(text, lang=lang))
        return [(w.text, w.phonemes) for w in sentence if w.phonemes]


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
