#!/usr/bin/env python3
"""Tests for SSML"""
import unittest

from gruut import sentences


class SSMLTestCase(unittest.TestCase):
    """Test cases for SSML"""

    def test_wikipedia_example(self):
        """Test SSML example from Wikipedia"""
        text = """<?xml version="1.0"?>
<speak xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:dc="http://purl.org/dc/elements/1.1/"
       version="1.0">
  <metadata>
    <dc:title xml:lang="en">Telephone Menu: Level 1</dc:title>
  </metadata>

  <p>
    <s xml:lang="en-US">
      <voice name="David" gender="male" age="25">
        For English, press <emphasis>one</emphasis>.
      </voice>
    </s>
    <s xml:lang="es-MX">
      <voice name="Miguel" gender="male" age="25">
        Para español, oprima el <emphasis>dos</emphasis>.
      </voice>
    </s>
  </p>

</speak>
"""

        results = [
            (w.sent_idx, w.idx, w.lang, w.voice, w.text)
            for sent in sentences(text, ssml=True)
            for w in sent
        ]

        self.assertEqual(
            results,
            [
                (0, 0, "en-US", "David", "For"),
                (0, 1, "en-US", "David", "English"),
                (0, 2, "en-US", "David", ","),
                (0, 3, "en-US", "David", "press"),
                (0, 4, "en-US", "David", "one"),
                (0, 5, "en-US", "David", "."),
                (1, 0, "es-MX", "Miguel", "Para"),
                (1, 1, "es-MX", "Miguel", "español"),
                (1, 2, "es-MX", "Miguel", ","),
                (1, 3, "es-MX", "Miguel", "oprima"),
                (1, 4, "es-MX", "Miguel", "el"),
                (1, 5, "es-MX", "Miguel", "dos"),
                (1, 6, "es-MX", "Miguel", "."),
            ],
        )


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
