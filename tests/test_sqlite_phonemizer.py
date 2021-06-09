#!/usr/bin/env python3
"""Tests for SqlitePhonemizer class"""
import unittest
from collections import defaultdict

from gruut import (
    SqlitePhonemizer,
    WordPronunciation,
    Token,
    TokenFeatures,
    UnknownWordError,
)
from gruut.lang import get_phonemizer


class SqlitePhonemizerTestCase(unittest.TestCase):
    """Test cases for SqlitePhonemizer class"""

    def test_crud(self):
        """Test database creation, insertion, and selection"""
        phonemizer = SqlitePhonemizer(database_path=":memory:")
        phonemizer.create_tables()

        test_prons = [
            WordPronunciation(["g", "ɹ", "u", "t"]),
            WordPronunciation(["g", "ɹ", "ʌ", "t"]),
            WordPronunciation(["g", "ɹ", "ʊ", "t"]),
        ]

        phonemizer.insert_prons("gruut", test_prons)

        actual_prons = list(pron for _word, pron in phonemizer.select_prons("gruut"))

        # Verify inserted pronunciations are returned
        self.assertEqual(test_prons, actual_prons)

        # Delete pronunciations
        phonemizer.delete_prons("gruut")

        actual_prons = list(pron for _word, pron in phonemizer.select_prons("gruut"))

        # Verify they were deleted
        self.assertEqual(len(actual_prons), 0)

    def test_select_prons(self):
        """Test different modes for select_prons"""
        phonemizer = SqlitePhonemizer(database_path=":memory:")
        phonemizer.create_tables()

        test_lexicon = {
            "this": [WordPronunciation(["ð", "ɪ", "s"])],
            "gruut": [
                WordPronunciation(["g", "ɹ", "u", "t"]),
                WordPronunciation(["g", "ɹ", "ʌ", "t"]),
            ],
            "test": [WordPronunciation(["t", "ɛ", "s", "t"])],
        }

        for word, word_prons in test_lexicon.items():
            phonemizer.insert_prons(word, word_prons)

        # Single word
        single_word = "gruut"
        single_lexicon = defaultdict(list)
        for word, word_pron in phonemizer.select_prons(single_word):
            single_lexicon[word].append(word_pron)

        self.assertEqual(single_lexicon, {single_word: test_lexicon[single_word]})

        # Multiple words
        multi_words = ["gruut", "test"]
        multi_lexicon = defaultdict(list)
        for word, word_pron in phonemizer.select_prons(multi_words):
            multi_lexicon[word].append(word_pron)

        self.assertEqual(multi_lexicon, {k: test_lexicon[k] for k in multi_words})

        # All words
        all_lexicon = defaultdict(list)
        for word, word_pron in phonemizer.select_prons():
            all_lexicon[word].append(word_pron)

        self.assertEqual(all_lexicon, test_lexicon)

    def test_word_index(self):
        """Test selection of pronunciation by index"""
        phonemizer = SqlitePhonemizer(database_path=":memory:", use_word_indexes=True)
        phonemizer.create_tables()

        test_prons = [
            WordPronunciation(["g", "ɹ", "u", "t"]),
            WordPronunciation(["g", "ɹ", "ʌ", "t"]),
        ]

        phonemizer.insert_prons("gruut", test_prons)

        actual_phonemes = next(phonemizer.phonemize([Token("gruut")]))

        # Verify first pronunciation was selected
        self.assertEqual(test_prons[0].phonemes, actual_phonemes)

        # Request second pronunciation
        actual_phonemes = next(phonemizer.phonemize([Token("gruut_2")]))

        # Verify second pronunciation was selected
        self.assertEqual(test_prons[1].phonemes, actual_phonemes)

    def test_insert_select_features(self):
        """Test insertion and selection of pronunciations with preferred features"""
        pos = TokenFeatures.PART_OF_SPEECH
        phonemizer = SqlitePhonemizer(database_path=":memory:", token_features=[pos])
        phonemizer.create_tables()

        # Distinguish wound (noun) from wound (verb)
        test_prons = [
            WordPronunciation(["w", "u", "n", "d"], preferred_features={pos: {"NN"}}),
            WordPronunciation(["w", "aʊ", "n", "d"], preferred_features={pos: {"VBD"}}),
        ]

        phonemizer.insert_prons("wound", test_prons)

        actual_prons = list(pron for _word, pron in phonemizer.select_prons("wound"))

        # Verify inserted pronunciations are returned
        self.assertEqual(test_prons, actual_prons)

    def test_phonemize_with_features(self):
        """Test selection of pronunciation using preferred features"""
        pos = TokenFeatures.PART_OF_SPEECH
        phonemizer = SqlitePhonemizer(database_path=":memory:", token_features=[pos])
        phonemizer.create_tables()

        # Distinguish wound (noun) from wound (verb)
        test_prons = [
            WordPronunciation(["w", "u", "n", "d"], preferred_features={pos: {"NN"}}),
            WordPronunciation(["w", "aʊ", "n", "d"], preferred_features={pos: {"VBD"}}),
        ]

        phonemizer.insert_prons("wound", test_prons)

        # First pronunciation (no features)
        actual_phonemes = next(phonemizer.phonemize([Token("wound")]))
        self.assertEqual(test_prons[0].phonemes, actual_phonemes)

        # Second pronunciation
        actual_phonemes = next(
            phonemizer.phonemize([Token("wound", features={pos: "VBD"})])
        )
        self.assertEqual(test_prons[1].phonemes, actual_phonemes)

        # First pronunciation again (with features)
        actual_phonemes = next(
            phonemizer.phonemize([Token("wound", features={pos: "NN"})])
        )
        self.assertEqual(test_prons[0].phonemes, actual_phonemes)

    def test_missing_word(self):
        """Test phonemize with a word not in the lexicon"""
        phonemizer = SqlitePhonemizer(database_path=":memory:")
        phonemizer.create_tables()

        # No guessing
        def empty_guess(token):
            return []

        phonemizer.guess_pronunciations = empty_guess

        # No phonemes
        actual_phonemes = next(phonemizer.phonemize([Token("missing")]))
        self.assertEqual(len(actual_phonemes), 0)

        # Add guessing
        missing_phonemes = ["m", "ɪ", "s", "ɪ", "ŋ"]

        def fixed_guess(token):
            if token.text == "missing":
                return [WordPronunciation(missing_phonemes)]

            return []

        phonemizer.guess_pronunciations = fixed_guess

        # "Guessed" phonemes
        actual_phonemes = next(phonemizer.phonemize([Token("missing")]))
        self.assertEqual(missing_phonemes, actual_phonemes)

    def test_missing_fail(self):
        """Test phonemize with a word not in the lexicon and failure is requested"""
        phonemizer = SqlitePhonemizer(
            database_path=":memory:", fail_on_unknown_words=True
        )
        phonemizer.create_tables()

        # No guessing
        def empty_guess(token):
            return []

        phonemizer.guess_pronunciations = empty_guess

        with self.assertRaises(UnknownWordError):
            next(phonemizer.phonemize([Token("missing")]))

    def test_non_word_chars(self):
        """Test lookup with non-word characters removed"""
        phonemizer = SqlitePhonemizer(database_path=":memory:")
        phonemizer.create_tables()

        # No guessing
        def empty_guess(token):
            return []

        phonemizer.guess_pronunciations = empty_guess

        # Lack of apostrophe
        test_prons = [WordPronunciation(["d", "oʊ", "n", "t"])]
        phonemizer.insert_prons("dont", test_prons)

        # Should fail because "don't" with an apostrophe is not in the lexcion
        actual_phonemes = next(phonemizer.phonemize([Token("don't")]))
        self.assertEqual(len(actual_phonemes), 0)

        # Allow non-word characters to be stripped
        phonemizer.lookup_with_only_words_chars = True
        actual_phonemes = next(phonemizer.phonemize([Token("don't")]))

        # Succeeds now
        self.assertEqual(actual_phonemes, test_prons[0].phonemes)

    def test_word_breaks(self):
        """Test addition of break phonemes between words"""
        phonemizer = SqlitePhonemizer(database_path=":memory:", word_break="#")
        phonemizer.create_tables()

        # No guessing
        def empty_guess(token):
            return []

        phonemizer.guess_pronunciations = empty_guess

        phonemizer.insert_prons("this", [WordPronunciation(["ð", "ɪ", "s"])])
        phonemizer.insert_prons("is", [WordPronunciation(["ɪ", "z"])])
        phonemizer.insert_prons("a", [WordPronunciation(["ə"])])
        phonemizer.insert_prons("test", [WordPronunciation(["t", "ɛ", "s", "t"])])

        actual_phonemes = list(phonemizer.phonemize("this is a test".split()))

        # Break symbol (#) between all words (including bos/eos)
        self.assertEqual(
            actual_phonemes,
            [
                ["#"],
                ["ð", "ɪ", "s"],
                ["#"],
                ["ɪ", "z"],
                ["#"],
                ["ə"],
                ["#"],
                ["t", "ɛ", "s", "t"],
                ["#"],
            ],
        )

    def test_all_breaks(self):
        """Test addition of break phonemes between words with major/minor breaks"""
        phonemizer = SqlitePhonemizer(
            database_path=":memory:",
            word_break="#",
            major_breaks={".": "||"},
            minor_breaks={",": "|"},
        )
        phonemizer.create_tables()

        # No guessing
        def empty_guess(token):
            return []

        phonemizer.guess_pronunciations = empty_guess

        phonemizer.insert_prons("this", [WordPronunciation(["ð", "ɪ", "s"])])
        phonemizer.insert_prons("is", [WordPronunciation(["ɪ", "z"])])
        phonemizer.insert_prons("a", [WordPronunciation(["ə"])])
        phonemizer.insert_prons("test", [WordPronunciation(["t", "ɛ", "s", "t"])])

        actual_phonemes = list(phonemizer.phonemize("this , is a test .".split()))

        # Break symbol (#) between all words (including bos/eos).
        # Major/minor breaks are interspersed.
        # Note that word break does not occur after major break.
        self.assertEqual(
            actual_phonemes,
            [
                ["#"],
                ["ð", "ɪ", "s"],
                ["#"],
                ["|"],
                ["#"],
                ["ɪ", "z"],
                ["#"],
                ["ə"],
                ["#"],
                ["t", "ɛ", "s", "t"],
                ["#"],
                ["||"],
            ],
        )

    def test_feature_map(self):
        """Test grapheme to phoneme model for guessing pronunciations"""
        pos = TokenFeatures.PART_OF_SPEECH
        phonemizer = SqlitePhonemizer(
            database_path=":memory:",
            token_features=[pos],
            feature_map={pos: {"NN": "N", "VBD": "V"}},
        )
        phonemizer.create_tables()

        # Distinguish wound (noun) from wound (verb).
        # Using simplified features.
        test_prons = [
            WordPronunciation(["w", "u", "n", "d"], preferred_features={pos: {"N"}}),
            WordPronunciation(["w", "aʊ", "n", "d"], preferred_features={pos: {"V"}}),
        ]

        phonemizer.insert_prons("wound", test_prons)

        # First pronunciation (no features)
        actual_phonemes = next(phonemizer.phonemize([Token("wound")]))
        self.assertEqual(test_prons[0].phonemes, actual_phonemes)

        # Second pronunciation
        # VBD will be mapped to V with feature map.
        actual_phonemes = next(
            phonemizer.phonemize([Token("wound", features={pos: "VBD"})])
        )
        self.assertEqual(test_prons[1].phonemes, actual_phonemes)

        # First pronunciation again (with features)
        # NN will be mapped to N with feature map.
        actual_phonemes = next(
            phonemizer.phonemize([Token("wound", features={pos: "NN"})])
        )
        self.assertEqual(test_prons[0].phonemes, actual_phonemes)

    def test_preload_prons(self):
        """Test preload_prons method"""
        phonemizer = get_phonemizer("en-us")
        self.assertEqual(len(phonemizer.lexicon), 0)

        phonemizer.preload_prons()
        self.assertGreater(len(phonemizer.lexicon), 0)

    def test_clean_phonemes_stress(self):
        """Test removal of stress"""
        # Defaults to with stress
        phonemizer = get_phonemizer("en-us")
        actual_phonemes = next(phonemizer.phonemize([Token("test")]))
        self.assertEqual(actual_phonemes, ["t", "ˈɛ", "s", "t"])

        # Remove stress
        phonemizer = get_phonemizer("en-us", remove_stress=True)
        actual_phonemes = next(phonemizer.phonemize([Token("test")]))
        self.assertEqual(actual_phonemes, ["t", "ɛ", "s", "t"])

    def test_clean_phonemes_accents(self):
        """Test removal of accents"""
        # Defaults to with accents
        phonemizer = SqlitePhonemizer(database_path=":memory:")
        phonemizer.create_tables()

        accent_pron = ["²'ɑː", "b", "eː"]
        phonemizer.insert_prons("AB", [WordPronunciation(accent_pron)])

        actual_phonemes = next(phonemizer.phonemize([Token("AB")]))
        self.assertEqual(actual_phonemes, accent_pron)

        # Disable accents
        phonemizer = SqlitePhonemizer(database_path=":memory:", remove_accents=True)
        phonemizer.create_tables()

        accent_pron = ["²'ɑː", "b", "eː"]
        phonemizer.insert_prons("AB", [WordPronunciation(accent_pron)])

        actual_phonemes = next(phonemizer.phonemize([Token("AB")]))
        self.assertEqual(actual_phonemes, ["ɑː", "b", "eː"])


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
