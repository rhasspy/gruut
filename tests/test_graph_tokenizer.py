#!/usr/bin/env python3
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

        token_words = [t.text.strip() for t in sentence.tokens]
        self.assertEqual(expected_words, token_words)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
