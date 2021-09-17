#!/usr/bin/env python3
"""Tests for EnglishTokenizer class"""
import unittest

# from gruut.lang import get_tokenizer
from gruut import get_text_processor


class EnglishTokenizerTestCase(unittest.TestCase):
    """Test cases for EnglishTokenizer class"""

    def test_unclean_text(self):
        """Test text with lots of noise"""
        processor = get_text_processor("en_US")

        text = "IT’S <a> 'test' (seNtEnce) for $100, dr., & [ I ] ## like ## it 100%!"
        sentences = list(processor.sentences(*processor(text)))
        self.assertEqual(1, len(sentences))
        sentence = sentences[0]

        self.assertEqual(
            [
                "it's",
                "a",
                "test",
                "sentence",
                "for",
                "one",
                "hundred",
                "dollars",
                ",",
                "doctor",
                ",",
                "and",
                "i",
                "like",
                "it",
                "one",
                "hundred",
                "percent",
                "!",
            ],
            [t.text for t in sentence.tokens],
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
