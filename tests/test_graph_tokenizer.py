#!/usr/bin/env python3
import re
import unittest

from gruut.graph_tokenizer import GraphTokenizer


class GraphTokenizerTests(unittest.TestCase):
    def test_split_whitespace(self):
        """Test whitespace splitting"""
        tokenizer = GraphTokenizer()

        text = "This  is a test                   sentence.  "
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        # Verify whitespace is preseved exactly
        actual_text = "".join(t.text for t in sentences[0].tokens)

        self.assertEqual(text, actual_text)

    def test_replacements(self):
        """Test pre-tokenize replacements"""
        tokenizer = GraphTokenizer(replacements=[("before", "after replacement")])

        text = "This is before"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        expected_words = ["This", "is", "after", "replacement"]
        actual_words = [t.text.strip() for t in sentences[0].tokens]

        self.assertEqual(expected_words, actual_words)

    def test_punctuation(self):
        """Test with punctuation symbols"""
        tokenizer = GraphTokenizer(punctuations={"!", ",", ":", '"'})

        text = 'Hear this: a "test" sentence, but with punctuation!'
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        sentence = sentences[0]

        # Raw text and words should match original text
        self.assertEqual(text, sentence.raw_text)
        self.assertEqual(
            [w.strip() for w in sentence.raw_words],
            [
                "Hear",
                "this:",
                "a",
                '"test"',
                "sentence,",
                "but",
                "with",
                "punctuation!",
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

        self.assertEqual(expected_words, [w.strip() for w in sentence.clean_words])

    def test_abbreviations(self):
        """Test abbreviation expansion"""
        tokenizer = GraphTokenizer(
            # Need period as a break so it gets split during tokenization
            major_breaks={"."},
            # short_form -> [long, form]
            abbreviations={
                "((?P<d>[dD])r.)": [r"\g<d>octor"],
                "((?P<w>[wW])3(?P<c>[cC]))": r"\g<w>orld \g<w>ide \g<w>eb \g<c>onsortium".split(),
            },
        )

        text = "Dr. Jones was invited to the W3C."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        sentence = sentences[0]
        expected_words = [
            "Doctor",
            "Jones",
            "was",
            "invited",
            "to",
            "the",
            "World",
            "Wide",
            "Web",
            "Consortium",
        ]

        self.assertEqual(expected_words, [w.strip() for w in sentence.clean_words])

    def test_abbreviation_pattern(self):
        """Test abbreviation expansion using a custom pattern"""
        tokenizer = GraphTokenizer(
            minor_breaks={","},
            # short_form -> [long, form]
            abbreviations={re.compile(r"^([0-9]+)\s*$"): "<num1>", "(456)": "<num2>"},
        )

        # 123 will match <num1> pattern.
        # 456, will match <num2> pattern because it was surrounded by optional
        # punctuation.
        text = "123 456,"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))

        sentence = sentences[0]
        expected_words = ["<num1>", "<num2>"]

        self.assertEqual(expected_words, [w.strip() for w in sentence.clean_words])

    def test_breaks(self):
        """Test major/minor breaks"""
        tokenizer = GraphTokenizer(
            punctuations={'"'}, major_breaks={"."}, minor_breaks={","}
        )

        text = 'Major breaks separate sentences. Minor breaks, while "minor", are kept too.'
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(2, len(sentences))

        # Period at the end is kept
        self.assertEqual(
            ["Major", "breaks", "separate", "sentences", "."],
            [w.strip() for w in sentences[0].clean_words],
        )

        # Commas and period are kept, but quotes are not.
        self.assertEqual(
            ["Minor", "breaks", ",", "while", "minor", ",", "are", "kept", "too", "."],
            [w.strip() for w in sentences[1].clean_words],
        )

    def test_numbers_basic(self):
        """Test basic number to words expansion"""
        tokenizer = GraphTokenizer(num2words_lang="en_US")

        text = "The answer is 42"
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 42 -> forty two
        self.assertEqual(
            ["The", "answer", "is", "forty", "two"],
            [w.strip() for w in sentence.clean_words],
        )

        # Still digits in raw text
        self.assertEqual(
            ["The", "answer", "is", "42"], [w.strip() for w in sentence.raw_words]
        )

        # Disable number replacement
        sentences = list(tokenizer.tokenize(text, replace_numbers=False))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # Still digits now in clean text
        self.assertEqual(
            ["The", "answer", "is", "42"], [w.strip() for w in sentence.clean_words]
        )

    def test_real_numbers(self):
        """Test real number to words expansion"""
        tokenizer = GraphTokenizer(major_breaks={"."}, locale="en_US")

        text = "The answer is 4.2."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 4.2 -> four point two
        self.assertEqual(
            ["The", "answer", "is", "four", "point", "two", "."],
            [w.strip() for w in sentence.clean_words],
        )

        # Still digits in raw text
        self.assertEqual(
            ["The", "answer", "is", "4.2."], [w.strip() for w in sentence.raw_words]
        )

    def test_real_numbers_de(self):
        """Test real number to words expansion with German locale"""
        tokenizer = GraphTokenizer(
            major_breaks={"."}, minor_breaks={","}, locale="de_DE"
        )

        text = "Die Nummer ist 4,2."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # 4,2 -> vier Komma zwei
        self.assertEqual(
            ["Die", "Nummer", "ist", "vier", "Komma", "zwei", "."],
            [w.strip() for w in sentence.clean_words],
        )

        # Still digits in raw text
        self.assertEqual(
            ["Die", "Nummer", "ist", "4,2."], [w.strip() for w in sentence.raw_words]
        )

    def test_currency(self):
        """Test currency to words expansion"""
        tokenizer = GraphTokenizer(
            major_breaks={"."},
            # Map currency symbol to a name for num2words
            currencies={"$": "USD", "€": "EUR"},
            # Turn on currency replacement
            replace_currency=True,
            locale="en_US",
        )

        text = "That will cost $5."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # $5 -> five dolllars
        self.assertEqual(
            ["That", "will", "cost", "five", "dollars", "."],
            [w.strip() for w in sentence.clean_words],
        )

        # Still digits in raw text.
        self.assertEqual(
            ["That", "will", "cost", "$5."], [w.strip() for w in sentence.raw_words]
        )

        # Try with Euros
        text = "That will cost €5."
        sentences = list(tokenizer.tokenize(text))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        # €5 -> five euro
        self.assertEqual(
            ["That", "will", "cost", "five", "euro", "."],
            [w.strip() for w in sentence.clean_words],
        )

        # Still digits in raw text.
        self.assertEqual(
            ["That", "will", "cost", "€5."], [w.strip() for w in sentence.raw_words]
        )

    # def test_number_converter_ordinal(self):
    #     """Test ordinal number converter"""
    #     tokenizer = GraphTokenizer(locale="en_US", use_number_converters=True)

    #     text = "It's the 23_ordinal"
    #     sentences = list(tokenizer.tokenize(text))
    #     self.assertEqual(1, len(sentences))
    #     sentence = sentences[0]

    #     # 23_ordinal -> twenty third
    #     self.assertEqual(
    #         ["It's", "the", "twenty", "third"], [t.text for t in sentence.tokens]
    #     )

    #     # Still digits in raw text
    #     self.assertEqual(["It's", "the", "23_ordinal"], sentence.raw_words)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
