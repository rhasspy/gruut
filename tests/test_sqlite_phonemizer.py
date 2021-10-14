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
            get_phonemes("حَوّامتي مُمْتِلئة", "ar"),
            [
                ("حَوَّامَتُي", ["ħ", "a", "u", "aː", "m", "t", "iː"]),
                ("مُمْتِلِئَة", ["m", "u", "m", "t", "i", "l", "i", "ʔ", "i"],),
            ],
        )

    def test_cs(self):
        """Czech test"""
        self.assertEqual(
            get_phonemes("Moje vznášedlo je plné úhořů.", "cs-cz"),
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
            get_phonemes("Mein Luftkissenfahrzeug ist voller Aale.", "de_DE"),
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
            get_phonemes("My hovercraft is full of eels.", "en_US"),
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
            get_phonemes("Mi aerodeslizador está lleno de anguilas.", "es_ES"),
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

    def test_fa(self):
        """Farsi test"""
        self.assertEqual(
            get_phonemes("هاورکرافت من پر مارماهى است", "fa"),
            [
                (
                    "هاورکرافت",
                    ["h", "ɒː", "v", "æ", "ɾ", "k", "æ", "ɾ", "ɒː", "f", "t", "e̞"],
                ),
                ("من", ["m", "æ", "n"],),
                ("پر", ["p", "o", "ɾ"]),
                ("مارماهى", ["m", "ɒː", "ɾ", "æ", "m", "ɒː", "h", "e̞", "l"]),
                ("است", ["æ", "s", "t"]),
            ],
        )

    def test_fr(self):
        """French test"""
        self.assertEqual(
            get_phonemes("Mon aéroglisseur est plein d'anguilles.", "fr_FR"),
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
            get_phonemes("Il mio hovercraft è pieno di anguille.", "it_IT"),
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
            get_phonemes("Mijn luchtkussenboot zit vol paling.", "nl"),
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

    def test_pt(self):
        """Portuguese test"""
        self.assertEqual(
            get_phonemes("O meu hovercraft está cheio de enguias.", "pt"),
            [
                ("O", ["u"]),
                ("meu", ["m", "ew"],),
                ("hovercraft", ["o", "v", "e", "ɹ", "k", "ɾ", "ɐ", "f", "t", "ʃ"]),
                ("está", ["e", "s", "t", "ɐ"]),
                ("cheio", ["ʃ", "ej", "u"]),
                ("de", ["d", "ʒ", "i"]),
                ("enguias", ["ẽ", "ɡ", "j", "ɐ", "s"]),
                (".", ["‖"]),
            ],
        )

    def test_ru(self):
        """Russian test"""
        self.assertEqual(
            get_phonemes("Моё судно на воздушной подушке полно угрей.", "ru_RU"),
            [
                ("Моё", ["m", "o", "j", "oː"]),
                ("судно", ["s", "uː", "d", "n", "o"],),
                ("на", ["n", "aː"]),
                ("воздушной", ["v", "o", "z", "d", "uː", "ʂ", "n", "o", "j"]),
                ("подушке", ["p", "o", "d", "uː", "ʂ", "kʲ", "e"]),
                ("полно", ["p", "oː", "l", "n", "o"]),
                ("угрей", ["u", "ɡ", "rʲ", "eː", "j"]),
                (".", ["‖"]),
            ],
        )

    def test_sv(self):
        """Swedish test"""
        self.assertEqual(
            get_phonemes("Min svävare är full med ål.", "sv_SE"),
            [
                ("Min", ["m", "iː", "n"]),
                ("svävare", ["²s", "v", "'ɛː", "v", "a", "r", "ɛ"],),
                ("är", ["ɛː", "r"]),
                ("full", ["f", "ɵ", "l"]),
                ("med", ["m", "eː", "d"]),
                ("ål", ["oː", "l"]),
                (".", ["‖"]),
            ],
        )

    def test_sw(self):
        """Swahili test"""
        self.assertEqual(
            get_phonemes("Gari langu linaloangama limejaa na mikunga.", "sw"),
            [
                ("Gari", ["ɠ", "ɑ", "ɾ", "i"]),
                ("langu", ["l", "ɑ", "ᵑg", "u"],),
                (
                    "linaloangama",
                    ["l", "i", "n", "ɑ", "l", "ɔ", "ɑ", "ᵑg", "ɑ", "m", "ɑ"],
                ),
                ("limejaa", ["l", "i", "m", "ɛ", "ʄ", "ɑ", "ɑ"]),
                ("na", ["n", "ɑ"]),
                ("mikunga", ["m", "i", "k", "u", "ᵑg", "ɑ"]),
                (".", ["‖"]),
            ],
        )


def get_phonemes(text, lang):
    """Return (text, phonemes) for each word"""
    sentence = next(sentences(text, lang=lang))
    return [(w.text, w.phonemes) for w in sentence if w.phonemes]


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
