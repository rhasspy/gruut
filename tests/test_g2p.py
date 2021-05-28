#!/usr/bin/env python3
"""Tests for GraphemesToPhonemes class"""
import unittest

from gruut.g2p import GraphemesToPhonemes


class GraphemesToPhonemesTestCase(unittest.TestCase):
    """Test cases for GraphemesToPhonemes class"""

    def test_encode_decode(self):
        """Test encode/decode functions for pycrfsuite features"""
        s = "ði ıntəˈnæʃənəl fəˈnɛtık əsoʊsiˈeıʃn"
        self.assertEqual(
            GraphemesToPhonemes.decode_string(GraphemesToPhonemes.encode_string(s)), s
        )

    def test_features(self):
        """Test word features"""
        word = "test"
        actual_features = GraphemesToPhonemes.word2features(
            word, chars_forward=1, chars_backward=1, encode=False
        )

        expected_features = [
            {"bias": 1.0, "grapheme": "t", "begin": True, "grapheme+1": "e"},
            {"bias": 1.0, "grapheme": "e", "grapheme+1": "s", "grapheme-1": "t"},
            {"bias": 1.0, "grapheme": "s", "grapheme+1": "t", "grapheme-1": "e"},
            {"bias": 1.0, "grapheme": "t", "end": True, "grapheme-1": "s"},
        ]

        self.assertEqual(expected_features, actual_features)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
