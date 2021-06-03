#!/usr/bin/env python3
"""Tests for FrenchPhonemizer class"""
import typing
import unittest

from gruut import text_to_phonemes, Token
from gruut.lang import get_phonemizer, get_tokenizer

# # https://www.commeunefrancaise.com/blog/la-liaison

# # After a determiner
# Un enfant.
# Les arbres.
# Deux amis.
# Ton excellent vin.
# Ces autres voyages.

# # Pronoun + verb
# On est là!
# Elles ont faim!
# Vous êtes sûrs?
# Tu nous entends.
# Je les adore.

# # Adjective + noun
# J’ai des petites oreilles.
# Michel est un grand ami.
# Je regarde la télé sur un petit écran.
# C’est un ancien élève.

# # After short prepositions, and “très”
# C’est très amusant!
# Je vis en Amérique.
# Ils sont chez eux.
# J’arrive dans une minute.

# # Others
# Un bâtiment est en vue de l’île.
# Sa vie n’était pas en danger.
# Amalia est en danger.
# C`est incroyable!

class FrenchPhonemizerTestCase(unittest.TestCase):
    """Test cases for FrenchPhonemizer class"""

    def test_liason(self):
        """Test addition of liason"""
        tokenizer = get_tokenizer("fr-fr")

        # After a determiner
        sentence = self._without_and_with_liason(
            "Les arbres", "les", ["l", "e"], ["l", "e", "z"]
        )

        # Pronoun + verb
        sentence = self._without_and_with_liason(
            "On est là!", "on", ["ɔ̃"], ["ɔ̃", "n"]
        )

        # Adjective + noun
        sentence = self._without_and_with_liason(
            "J’ai des petites oreilles.",
            "petites",
            ["p", "ə", "t", "i", "t"],
            ["p", "ə", "t", "i", "t", "z"],
        )

        # After short prepositions, and “très”
        sentence = self._without_and_with_liason(
            "C’est très amusant!", "très", ["t", "ʁ", "ɛ"], ["t", "ʁ", "ɛ", "z"]
        )

    def test_last_token(self):
        """Ensure liason does not leave last token"""
        phonemes = text_to_phonemes("Est-ce-que", lang="fr")
        self.assertGreater(len(phonemes), 0)

    def _without_and_with_liason(
        self,
        text: str,
        liason_word: str,
        without_phonemes: typing.List[str],
        with_phonemes: typing.List[str],
    ):
        """Get pronunciation of a sentence with and without liason enabled"""
        tokenizer = get_tokenizer("fr-fr")
        sentence = next(tokenizer.tokenize(text))

        self.assertIn(liason_word, sentence.clean_words)

        # Verify no liason
        phonemizer_no_liason = get_phonemizer("fr-fr", fr_no_liason=True)
        phonemes_no_liason = list(phonemizer_no_liason.phonemize(sentence.tokens))
        for word, word_phonemes in zip(sentence.clean_words, phonemes_no_liason):
            if word == liason_word:
                self.assertEqual(word_phonemes, without_phonemes)

        # Verify liason
        phonemizer_liason = get_phonemizer("fr-fr")
        phonemes_liason = list(phonemizer_liason.phonemize(sentence.tokens))

        for word, word_phonemes in zip(sentence.clean_words, phonemes_liason):
            if word == liason_word:
                self.assertEqual(word_phonemes, with_phonemes)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
