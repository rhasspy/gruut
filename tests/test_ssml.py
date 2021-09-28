#!/usr/bin/env python3
"""Tests for SSML"""
import sys
import unittest

from gruut import sentences
from gruut.utils import print_graph


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

</speak>"""

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

    def test_lang_s(self):
        """Test lang on <s>"""
        text = """<?xml version="1.0" encoding="ISO-8859-1"?>
    <speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
        xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                    http://www.w3.org/TR/speech-synthesis11/synthesis.xsd"
        xml:lang="en-US">
    <s>Today, 2/1/2000.</s>
    <!-- Today, February first two thousand -->
    <s xml:lang="it">Un mese fà, 2/1/2000.</s>
    <!-- Un mese fà, il due gennaio duemila -->
    <!-- One month ago, the second of January two thousand -->
    </speak>"""

        results = [
            (w.sent_idx, w.idx, w.lang, w.text)
            for sent in sentences(text, ssml=True)
            for w in sent
        ]

        self.assertEqual(
            results,
            [
                (0, 0, "en-US", "Today"),
                (0, 1, "en-US", ","),
                (0, 2, "en-US", "February"),
                (0, 3, "en-US", "first"),
                (0, 4, "en-US", ","),
                (0, 5, "en-US", "two"),
                (0, 6, "en-US", "thousand"),
                (0, 7, "en-US", "."),
                (1, 0, "it", "Un"),
                (1, 1, "it", "mese"),
                (1, 2, "it", "fà"),
                (1, 3, "it", ","),
                # no "il"
                (1, 4, "it", "due"),
                (1, 5, "it", "gennaio"),
                (1, 6, "it", "duemila"),
                (1, 7, "it", "."),
            ],
        )

    def test_phoneme(self):
        """Test manual phoneme insertion"""
        text = """<?xml version="1.0"?>
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                 http://www.w3.org/TR/speech-synthesis11/synthesis.xsd"
       xml:lang="en-US">
  <phoneme alphabet="ipa" ph="t&#x259;mei&#x325;&#x27E;ou&#x325;"> tomato </phoneme>
  <!-- This is an example of IPA using character entities -->
  <!-- Because many platform/browser/text editor combinations do not
       correctly cut and paste Unicode text, this example uses the entity
       escape versions of the IPA characters.  Normally, one would directly
       use the UTF-8 representation of these symbols: "təmei̥ɾou̥". -->
</speak>"""

        results = [
            (w.sent_idx, w.idx, w.lang, w.text, w.phonemes)
            for sent in sentences(text, ssml=True)
            for w in sent
        ]

        self.assertEqual(
            results,
            [(0, 0, "en-US", "tomato", ["t", "ə", "m", "e", "i̥", "ɾ", "o", "u̥"])],
        )

    def test_sentences(self):
        """Test <s>"""
        text = """<?xml version="1.0"?>
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                 http://www.w3.org/TR/speech-synthesis11/synthesis.xsd"
       xml:lang="en-US">
  <p>
    <s>This is the first sentence of the paragraph.</s>
    <s>Here's another sentence.</s>
  </p>
</speak>"""

        results = [
            (w.sent_idx, w.idx, w.text)
            for sent in sentences(text, ssml=True)
            for w in sent
        ]

        self.assertEqual(
            results,
            [
                (0, 0, "This"),
                (0, 1, "is"),
                (0, 2, "the"),
                (0, 3, "first"),
                (0, 4, "sentence"),
                (0, 5, "of"),
                (0, 6, "the"),
                (0, 7, "paragraph"),
                (0, 8, "."),
                (1, 0, "Here's"),
                (1, 1, "another"),
                (1, 2, "sentence"),
                (1, 3, "."),
            ],
        )

    def test_token(self):
        """Test explicit tokenization"""

        # NOTE: Added full stops
        text = """<?xml version="1.0"?>
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                 http://www.w3.org/TR/speech-synthesis11/synthesis.xsd"
       xml:lang="zh-CN">

  <!-- The Nanjing Changjiang River Bridge -->
  <token>南京市</token><token>长江大桥</token>。
  <!-- The mayor of Nanjing city, Jiang Daqiao -->
  南京市长<w>江大桥</w>。
  <!-- Shanghai is a metropolis -->
  上海是个<w>大都会</w>。
  <!-- Most Shanghainese will say something like that -->
  上海人<w>大都</w>会那么说。
</speak>"""

        results = [
            (w.sent_idx, w.idx, w.text)
            for sent in sentences(text, ssml=True)
            for w in sent
        ]

        self.assertEqual(
            results,
            [
                (0, 0, "南京市"),
                (0, 1, "长江大桥"),
                (0, 2, "。"),
                (1, 0, "南"),
                (1, 1, "京"),
                (1, 2, "市"),
                (1, 3, "长"),
                (1, 4, "江大桥"),
                (1, 5, "。"),
                (2, 0, "上"),
                (2, 1, "海"),
                (2, 2, "是"),
                (2, 3, "个"),
                (2, 4, "大都会"),
                (2, 5, "。"),
                (3, 0, "上"),
                (3, 1, "海"),
                (3, 2, "人"),
                (3, 3, "大都"),
                (3, 4, "会"),
                (3, 5, "那"),
                (3, 6, "么"),
                (3, 7, "说"),
                (3, 8, "。"),
            ],
        )

    def test_sub(self):
        """Test <sub>"""
        text = """<?xml version="1.0"?>
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                 http://www.w3.org/TR/speech-synthesis11/synthesis.xsd"
       xml:lang="en-US">
  <sub alias="World Wide Web Consortium">W3C</sub>
  <!-- World Wide Web Consortium -->
</speak>"""

        results = [
            (w.sent_idx, w.idx, w.text)
            for sent in sentences(text, ssml=True)
            for w in sent
        ]

        self.assertEqual(
            results,
            [(0, 0, "World"), (0, 1, "Wide"), (0, 2, "Web"), (0, 3, "Consortium")],
        )

    def test_lang_element(self):
        """Test <lang>"""
        text = """<?xml version="1.0"?>
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                 http://www.w3.org/TR/speech-synthesis11/synthesis.xsd"
       xml:lang="en-US">
  The French word for cat is <w xml:lang="fr">chat</w>.
  He prefers to eat pasta that is <lang xml:lang="it">al dente</lang>.
</speak>"""

        results = [
            (w.sent_idx, w.idx, w.lang, w.text)
            for sent in sentences(text, ssml=True)
            for w in sent
        ]

        self.assertEqual(
            results,
            [
                (0, 0, "en-US", "The"),
                (0, 1, "en-US", "French"),
                (0, 2, "en-US", "word"),
                (0, 3, "en-US", "for"),
                (0, 4, "en-US", "cat"),
                (0, 5, "en-US", "is"),
                (0, 6, "fr", "chat"),
                (0, 7, "en-US", "."),
                (1, 0, "en-US", "He"),
                (1, 1, "en-US", "prefers"),
                (1, 2, "en-US", "to"),
                (1, 3, "en-US", "eat"),
                (1, 4, "en-US", "pasta"),
                (1, 5, "en-US", "that"),
                (1, 6, "en-US", "is"),
                (1, 7, "it", "al"),
                (1, 8, "it", "dente"),
                (1, 9, "en-US", "."),
            ],
        )


def print_graph_stderr(graph, root):
    """Print graph to stderr"""
    print_graph(graph, root, print_func=lambda *p: print(*p, file=sys.stderr))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
