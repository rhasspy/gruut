#!/usr/bin/env python3
"""Tests for PartOfSpeechTagger class"""
import copy
import unittest

from gruut.pos import PartOfSpeechTagger


class PartOfSpeechTaggerTestCase(unittest.TestCase):
    """Test cases for PartOfSpeechTagger class"""

    def test_encode_decode(self):
        """Test encode/decode functions for pycrfsuite features"""
        s = "ði ıntəˈnæʃənəl fəˈnɛtık əsoʊsiˈeıʃn"
        self.assertEqual(
            PartOfSpeechTagger.decode_string(PartOfSpeechTagger.encode_string(s)), s
        )

    def test_features(self):
        """Test sentence features"""
        sentence = "1 test .".split()
        word_features = {
            "1": {
                "bias": 1.0,
                "word": "1",
                "len(word)": 1,
                "word.ispunctuation": False,
                "word.isdigit()": True,
                "word[:2]": "1",
                "word[-2:]": "1",
            },
            "test": {
                "bias": 1.0,
                "word": "test",
                "len(word)": 4,
                "word.ispunctuation": False,
                "word[-2:]": "st",
                "word[:2]": "te",
                "word.isdigit()": False,
            },
            ".": {
                "bias": 1.0,
                "word": ".",
                "len(word)": 1,
                "word.ispunctuation": True,
                "word.isdigit()": False,
                "word[-2:]": ".",
                "word[:2]": ".",
            },
        }

        def add_prefix(d, prefix):
            return {f"{prefix}{k}": v for k, v in d.items()}

        # Add context
        context_features = copy.deepcopy(word_features)
        context_features["1"].update(add_prefix(word_features["test"], "+1:"))

        context_features["test"].update(add_prefix(word_features["1"], "-1:"))
        context_features["test"].update(add_prefix(word_features["."], "+1:"))

        context_features["."].update(add_prefix(word_features["test"], "-1:"))

        # Add BOS/EOS
        context_features["1"]["BOS"] = True
        context_features["."]["EOS"] = True

        expected_features = [
            context_features["1"],
            context_features["test"],
            context_features["."],
        ]

        actual_features = PartOfSpeechTagger.sent2features(
            sentence,
            words_forward=1,
            words_backward=1,
            chars_front=2,
            chars_back=2,
            encode=False,
        )

        self.assertEqual(expected_features, actual_features)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
