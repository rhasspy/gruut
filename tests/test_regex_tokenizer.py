#!/usr/bin/env python3
"""Tests for RegexTokenizer class"""
import pickle
import re
import unittest

from gruut import RegexTokenizer


class RegexTokenizerTestCase(unittest.TestCase):
    """Test cases for RegexTokenizer class"""

    def test_split_whitespace(self):
        """Test whitespace splitting"""
        tokenizer = RegexTokenizer()

        text = "This is a test sentence."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        # Should be split by just whitespace
        expected_words = text.split()
        actual_words = [t.text for t in sentences[0].tokens]

        self.assertEqual(expected_words, actual_words)

    def test_replacements(self):
        """Test pre-tokenize replacements"""
        tokenizer = RegexTokenizer(replacements=[("before", "after replacement")])

        text = "This is before"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        expected_words = ["This", "is", "after", "replacement"]
        actual_words = [t.text for t in sentences[0].tokens]

        self.assertEqual(expected_words, actual_words)

    def test_punctuation(self):
        """Test with punctuation symbols"""
        tokenizer = RegexTokenizer(punctuations={"!", ",", ":", '"'})

        text = 'Hear this: a "test" sentence, but with punctuation!'
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        sentence = sentences[0]

        # Raw text and words should match original text
        self.assertEqual(text, sentence.raw_text)
        self.assertEqual(
            sentence.raw_words,
            [
                "Hear",
                "this",
                ":",
                "a",
                '"',
                "test",
                '"',
                "sentence",
                ",",
                "but",
                "with",
                "punctuation",
                "!",
            ],
        )

        # Punctuation is removed from tokens and "clean" words unless it is a
        # major/minor break.
        expected_words = [
            "Hear",
            "this",
            "a",
            "test",
            "sentence",
            "but",
            "with",
            "punctuation",
        ]

        self.assertEqual(expected_words, sentence.clean_words)

        token_words = [t.text for t in sentence.tokens]
        self.assertEqual(expected_words, token_words)

    def test_abbreviations(self):
        """Test abbreviation expansion"""
        tokenizer = RegexTokenizer(
            # Need period as a punctuation so it gets split during tokenization
            punctuations={"."},
            # short_form -> [long, form]
            abbreviations={"dr.": "doctor", "w3c": "world wide web consortium"},
            # Apply a casing transformation so we only need to write one
            # abbreviation form.
            casing_func=str.lower,
        )

        text = "Dr. Jones was invited to the W3C."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        sentence = sentences[0]
        expected_words = [
            "doctor",
            "jones",
            "was",
            "invited",
            "to",
            "the",
            "world",
            "wide",
            "web",
            "consortium",
        ]

        self.assertEqual(expected_words, [t.text for t in sentence.tokens])

    def test_abbreviation_pattern(self):
        """Test abbreviation expansion using a custom pattern"""
        tokenizer = RegexTokenizer(
            punctuations={","},
            # short_form -> [long, form]
            abbreviations={re.compile(r"^([0-9]+)$"): "<num1>", "456": "<num2>"},
        )

        # 123 will match <num1> pattern.
        # 456, will match <num2> pattern because it was surrounded by optional
        # punctuation.
        text = "123 456,"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        sentence = sentences[0]
        expected_words = ["<num1>", "<num2>"]

        self.assertEqual(expected_words, [t.text for t in sentence.tokens])

    def test_breaks(self):
        """Test major/minor breaks"""
        tokenizer = RegexTokenizer(
            punctuations={".", ",", '"'}, major_breaks={"."}, minor_breaks={","}
        )

        text = 'Major breaks separate sentences. Minor breaks, while "minor", are kept too.'
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(2, len(sentences))

        # Period at the end is now kept, despite being punctuation.
        self.assertEqual(
            ["Major", "breaks", "separate", "sentences", "."],
            [t.text for t in sentences[0].tokens],
        )

        # Commas and period are kept, but quotes are not.
        self.assertEqual(
            ["Minor", "breaks", ",", "while", "minor", ",", "are", "kept", "too", "."],
            [t.text for t in sentences[1].tokens],
        )

    def test_numbers_basic(self):
        """Test basic number to words expansion"""
        tokenizer = RegexTokenizer(num2words_lang="en_US")

        text = "The answer is 42"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 42 -> forty two
        self.assertEqual(
            ["The", "answer", "is", "forty", "two"], [t.text for t in sentence.tokens]
        )

        # Still digits in raw text
        self.assertEqual(["The", "answer", "is", "42"], sentence.raw_words)

    def test_real_numbers_no_babel(self):
        """Test real number to words expansion without Babel"""
        tokenizer = RegexTokenizer(punctuations={"."}, num2words_lang="en_US")

        text = "The answer is 4.2."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 4.2 -> four point two
        self.assertEqual(
            ["The", "answer", "is", "four", "point", "two"],
            [t.text for t in sentence.tokens],
        )

        # Still digits in raw text
        self.assertEqual(["The", "answer", "is", "4.2", "."], sentence.raw_words)

    def test_real_numbers_babel(self):
        """Test real number to words expansion with Babel"""
        tokenizer = RegexTokenizer(
            punctuations={".", ","}, num2words_lang="de_DE", babel_locale="de_DE"
        )

        text = "Die Nummer ist 4,2."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 4,2 -> vier Komma zwei
        self.assertEqual(
            ["Die", "Nummer", "ist", "vier", "Komma", "zwei"],
            [t.text for t in sentence.tokens],
        )

        # Still digits in raw text
        self.assertEqual(["Die", "Nummer", "ist", "4,2", "."], sentence.raw_words)

    def test_currency_euro(self):
        """Test currency to words expansion Euros"""
        tokenizer = RegexTokenizer(
            punctuations={"."},
            # Map currency symbol to a name for num2words
            currency_names={"€": "EUR"},
            # Turn on currency replacement
            do_replace_currency=True,
            num2words_lang="en_US",
        )

        text = "That will cost €5"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # €5 -> five euro
        self.assertEqual(
            ["That", "will", "cost", "five", "euro"], [t.text for t in sentence.tokens]
        )

        # Still digits in raw text.
        # Note that the currency symbol has been split out.
        self.assertEqual(["That", "will", "cost", "€", "5"], sentence.raw_words)

    def test_number_converter_ordinal(self):
        """Test ordinal number converter"""
        tokenizer = RegexTokenizer(num2words_lang="en_US", use_number_converters=True)

        text = "It's the 23_ordinal"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 23_ordinal -> twenty third
        self.assertEqual(
            ["It's", "the", "twenty", "third"], [t.text for t in sentence.tokens]
        )

        # Still digits in raw text
        self.assertEqual(["It's", "the", "23_ordinal"], sentence.raw_words)

    def test_number_converter_year(self):
        """Test year number converter"""
        tokenizer = RegexTokenizer(num2words_lang="en_US", use_number_converters=True)

        text = "It's 2021_year"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 2021_year -> twenty twenty one
        self.assertEqual(
            ["It's", "twenty", "twenty", "one"], [t.text for t in sentence.tokens]
        )

        # Still digits in raw text
        self.assertEqual(["It's", "2021_year"], sentence.raw_words)

    def test_pickle(self):
        """Test that tokenizer is picklable"""
        expected_tokenizer = RegexTokenizer()
        data = pickle.dumps(expected_tokenizer)

        actual_tokenizer = pickle.loads(data)

        self.assertEqual(expected_tokenizer.__dict__, actual_tokenizer.__dict__)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
