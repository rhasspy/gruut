#!/usr/bin/env python3
"""Tests for English class"""
import unittest

from gruut import sentences


class EnglishTestCase(unittest.TestCase):
    """Test cases for English"""

    def test_unclean_text(self):
        """Test text with lots of noise"""
        text = (
            "IT’S <a> 'test' (seNtEnce) for-only $100, Dr., & [I] ## *like* ## it 100%!"
        )
        sentence = next(sentences(text, lang="en_US"))

        self.assertEqual(
            [
                "IT'S",
                "<",
                "a",
                ">",
                "'",
                "test",
                "'",
                "(",
                "seNtEnce",
                ")",
                "for",
                "only",
                "one",
                "hundred",
                "dollars",
                ",",
                "Doctor",
                ",",
                "and",
                "[",
                "I",
                "]",
                "*",
                "like",
                "*",
                "it",
                "one",
                "hundred",
                "percent",
                "!",
            ],
            [word.text for word in sentence],
        )

    def test_spell_out(self):
        """Test spell-out in say-as SSML tag"""
        text = '<say-as interpret-as="spell-out">abc@1+2-3*.*</say-as>'
        sentence = next(sentences(text, lang="en_US", ssml=True))

        self.assertEqual(
            [
                "a",
                "b",
                "c",
                "at",
                "one",
                "plus",
                "two",
                "dash",
                "three",
                "star",
                "dot",
                "star",
            ],
            [word.text for word in sentence],
        )

    def test_initialisms(self):
        """Test expansion of initialisms"""
        text = "ABC abc A.B.C."
        sentence = next(sentences(text, lang="en_US"))

        self.assertEqual(
            ["A", "B", "C", "abc", "A", "B", "C"], [word.text for word in sentence],
        )

    def test_dates(self):
        """Test expansion of dates"""
        text = "1/4/1999 vs. 4/1/1999"
        sentence = next(sentences(text, lang="en_US"))

        self.assertEqual(
            [
                "January",
                "fourth",
                ",",
                "nineteen",
                "ninety",
                "nine",
                "versus",
                "April",
                "first",
                ",",
                "nineteen",
                "ninety",
                "nine",
            ],
            [word.text for word in sentence],
        )

    def test_ordinals(self):
        """Test parsing of ordinal numbers"""
        text = "1st, 2nd, 3rd, 4th, 5th, 23rd, 32nd, 44th, 121st, 5,111st."
        sentence = next(sentences(text, lang="en_US"))

        self.assertEqual(
            [
                "first",
                ",",
                "second",
                ",",
                "third",
                ",",
                "fourth",
                ",",
                "fifth",
                ",",
                "twenty",
                "third",
                ",",
                "thirty",
                "second",
                ",",
                "forty",
                "fourth",
                ",",
                "one",
                "hundred",
                "and",
                "twenty",
                "first",
                ",",
                "five",
                "thousand",
                ",",
                "one",
                "hundred",
                "and",
                "eleventh",
                ".",
            ],
            [word.text for word in sentence],
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
