#!/usr/bin/env python3
"""Tests for French"""
import typing
import unittest

from gruut import sentences

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


class FrenchTestCase(unittest.TestCase):
    """Test cases for French"""

    def test_liason_after_determiner(self):
        """Test liason after determiner (e.g., le)"""
        self._without_and_with_liason("Les arbres", "Les", ["l", "e"], ["l", "e", "z"])

    def test_liason_adjective_noun(self):
        """Test liason between adjective and noun"""
        self._without_and_with_liason(
            "J’ai des petites oreilles.",
            "petites",
            ["p", "ə", "t", "i", "t"],
            ["p", "ə", "t", "i", "t", "z"],
        )

    def test_liason_pronoun_verb(self):
        """Test liason between pronoun and verb"""
        self._without_and_with_liason("On est là!", "On", ["ɔ̃"], ["ɔ̃", "n"])

    def test_liason_tres(self):
        """Test liason with très"""
        self._without_and_with_liason(
            "C’est très amusant!", "très", ["t", "ʁ", "ɛ"], ["t", "ʁ", "ɛ", "z"]
        )

    def _without_and_with_liason(
        self,
        text: str,
        liason_word: str,
        without_phonemes: typing.List[str],
        with_phonemes: typing.List[str],
    ):
        """Get pronunciation of a sentence with and without liason enabled"""

        # Verify no liason
        sentence = next(iter(sentences(text, lang="fr_FR", post_process=False)))
        word = next(w for w in sentence if w.text == liason_word)
        self.assertEqual(word.phonemes, without_phonemes)

        # Verify liason
        sentence = next(iter(sentences(text, lang="fr_FR", post_process=True)))
        word = next(w for w in sentence if w.text == liason_word)
        self.assertEqual(word.phonemes, with_phonemes)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
