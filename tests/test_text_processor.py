#!/usr/bin/env python3
"""Tests for TextProcessor"""
import sys
import unittest

from gruut.text_processor import Sentence, TextProcessor, TextProcessorSettings, Word
from gruut.utils import print_graph

WORDS_KWARGS = {"explicit_lang": False, "phonemes": False, "pos": False}


class TextProcessorTestCase(unittest.TestCase):
    """Tests for TextProcessor"""

    def test_whitespace(self):
        """Text whitespace preservation"""
        processor = TextProcessor()
        graph, root = processor("This is  a   test    ")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Whitespace is retained by default
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="This", text_with_ws="This "),
                Word(idx=1, sent_idx=0, text="is", text_with_ws="is  "),
                Word(idx=2, sent_idx=0, text="a", text_with_ws="a   "),
                Word(idx=3, sent_idx=0, text="test", text_with_ws="test    "),
            ],
        )

    def test_no_whitespace(self):
        """Test disabling of whitespace preservation"""
        processor = TextProcessor(keep_whitespace=False)
        graph, root = processor("This is  a   test    ")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Whitespace is discarded
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="This", text_with_ws="This"),
                Word(idx=1, sent_idx=0, text="is", text_with_ws="is"),
                Word(idx=2, sent_idx=0, text="a", text_with_ws="a"),
                Word(idx=3, sent_idx=0, text="test", text_with_ws="test"),
            ],
        )

    def test_punctuation(self):
        """Test splitting of punctuation from around words"""
        processor = TextProcessor(
            begin_punctuations={'"', "«"},
            end_punctuations={'"', "»"},
            minor_breaks={","},
            major_breaks={"."},
        )
        graph, root = processor('This «is»,  a "test".')
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Punctuations are separated
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="This", text_with_ws="This "),
                Word(
                    idx=1, sent_idx=0, text="«", text_with_ws="«", is_punctuation=True
                ),
                Word(idx=2, sent_idx=0, text="is", text_with_ws="is"),
                Word(
                    idx=3, sent_idx=0, text="»", text_with_ws="»", is_punctuation=True
                ),
                Word(
                    idx=4, sent_idx=0, text=",", text_with_ws=",  ", is_minor_break=True
                ),
                Word(idx=5, sent_idx=0, text="a", text_with_ws="a "),
                Word(
                    idx=6, sent_idx=0, text='"', text_with_ws='"', is_punctuation=True
                ),
                Word(idx=7, sent_idx=0, text="test", text_with_ws="test"),
                Word(
                    idx=8, sent_idx=0, text='"', text_with_ws='"', is_punctuation=True
                ),
                Word(
                    idx=9, sent_idx=0, text=".", text_with_ws=".", is_major_break=True
                ),
            ],
        )

    def test_punctuation_with_inner_break(self):
        """Test break inside of punctuation"""
        processor = TextProcessor(
            begin_punctuations={'"'}, end_punctuations={'"'}, major_breaks={"."},
        )
        graph, root = processor('Test "one." Test two.')
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # First sentence includes final quote
        self.assertEqual(
            words,
            [
                # First sentence
                Word(idx=0, sent_idx=0, text="Test", text_with_ws="Test "),
                Word(
                    idx=1, sent_idx=0, text='"', text_with_ws='"', is_punctuation=True
                ),
                Word(idx=2, sent_idx=0, text="one", text_with_ws="one"),
                Word(
                    idx=3, sent_idx=0, text=".", text_with_ws=".", is_major_break=True
                ),
                Word(
                    idx=4, sent_idx=0, text='"', text_with_ws='" ', is_punctuation=True
                ),
                # Second sentence
                Word(idx=0, sent_idx=1, text="Test", text_with_ws="Test "),
                Word(idx=1, sent_idx=1, text="two", text_with_ws="two"),
                Word(
                    idx=2, sent_idx=1, text=".", text_with_ws=".", is_major_break=True
                ),
            ],
        )

    def test_replacements(self):
        """Test regex replacements during tokenization"""
        processor = TextProcessor(
            minor_breaks={","},
            major_breaks={"."},
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
            ],
        )
        graph, root = processor("\"This,\" [is] <a> (test) 'sentence.'")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Quotes and brackets are discarded
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="This", text_with_ws="This"),
                Word(
                    idx=1, sent_idx=0, text=",", text_with_ws=", ", is_minor_break=True
                ),
                Word(idx=2, sent_idx=0, text="is", text_with_ws="is "),
                Word(idx=3, sent_idx=0, text="a", text_with_ws="a "),
                Word(idx=4, sent_idx=0, text="test", text_with_ws="test "),
                Word(idx=5, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(
                    idx=6, sent_idx=0, text=".", text_with_ws=".", is_major_break=True
                ),
            ],
        )

    def test_abbreviations(self):
        """Test expansion of abbreviations (with case preservation)"""
        processor = TextProcessor(
            minor_breaks={","},
            major_breaks={".", "?"},
            abbreviations={
                r"^([dD])r\.": r"\1octor",
                r"^([mM])r\.": r"\1ister",
                r"^([sS])t\.": r"\1treet",
            },
        )
        graph, root = processor("Mr.? I'm just a dr., on this St. at least.")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Abbreviations are expanded, maintaining capitalization
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="Mister", text_with_ws="Mister"),
                Word(
                    idx=1, sent_idx=0, text="?", text_with_ws="? ", is_major_break=True
                ),
                Word(idx=0, sent_idx=1, text="I'm", text_with_ws="I'm "),
                Word(idx=1, sent_idx=1, text="just", text_with_ws="just "),
                Word(idx=2, sent_idx=1, text="a", text_with_ws="a "),
                Word(idx=3, sent_idx=1, text="doctor", text_with_ws="doctor"),
                Word(
                    idx=4, sent_idx=1, text=",", text_with_ws=", ", is_minor_break=True
                ),
                Word(idx=5, sent_idx=1, text="on", text_with_ws="on "),
                Word(idx=6, sent_idx=1, text="this", text_with_ws="this "),
                Word(idx=7, sent_idx=1, text="Street", text_with_ws="Street "),
                Word(idx=8, sent_idx=1, text="at", text_with_ws="at "),
                Word(idx=9, sent_idx=1, text="least", text_with_ws="least"),
                Word(
                    idx=10, sent_idx=1, text=".", text_with_ws=".", is_major_break=True
                ),
            ],
        )

    def test_multiple_sentences(self):
        """Test sentence break"""
        processor = TextProcessor(major_breaks={".", "!"})
        graph, root = processor("First  sentence. Second sentence! ")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Separated by a major break
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="First", text_with_ws="First  "),
                Word(idx=1, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(
                    idx=2, sent_idx=0, text=".", text_with_ws=". ", is_major_break=True
                ),
                Word(idx=0, sent_idx=1, text="Second", text_with_ws="Second "),
                Word(idx=1, sent_idx=1, text="sentence", text_with_ws="sentence"),
                Word(
                    idx=2, sent_idx=1, text="!", text_with_ws="! ", is_major_break=True
                ),
            ],
        )

        # Check sentences too
        sentences = list(processor.sentences(graph, root, **WORDS_KWARGS))
        self.assertEqual(
            sentences,
            [
                Sentence(
                    idx=0,
                    text="First sentence.",
                    text_with_ws="First  sentence. ",
                    text_spoken="First sentence",
                    words=[
                        Word(idx=0, sent_idx=0, text="First", text_with_ws="First  "),
                        Word(
                            idx=1, sent_idx=0, text="sentence", text_with_ws="sentence"
                        ),
                        Word(
                            idx=2,
                            sent_idx=0,
                            text=".",
                            text_with_ws=". ",
                            is_major_break=True,
                        ),
                    ],
                ),
                Sentence(
                    idx=1,
                    text="Second sentence!",
                    text_with_ws="Second sentence! ",
                    text_spoken="Second sentence",
                    words=[
                        Word(idx=0, sent_idx=1, text="Second", text_with_ws="Second "),
                        Word(
                            idx=1, sent_idx=1, text="sentence", text_with_ws="sentence"
                        ),
                        Word(
                            idx=2,
                            sent_idx=1,
                            text="!",
                            text_with_ws="! ",
                            is_major_break=True,
                        ),
                    ],
                ),
            ],
        )

    def test_multiple_paragraphs(self):
        """Test paragraph index"""
        processor = TextProcessor()
        graph, root = processor(
            "<speak><p>First paragraph</p><p>Second paragraph</p></speak>", ssml=True
        )
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Sentences/words should be in different paragraphs
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, par_idx=0, text="First", text_with_ws="First "),
                Word(
                    idx=1,
                    sent_idx=0,
                    par_idx=0,
                    text="paragraph",
                    text_with_ws="paragraph",
                ),
                Word(
                    idx=0, sent_idx=0, par_idx=1, text="Second", text_with_ws="Second "
                ),
                Word(
                    idx=1,
                    sent_idx=0,
                    par_idx=1,
                    text="paragraph",
                    text_with_ws="paragraph",
                ),
            ],
        )

    def test_explicit_sentence(self):
        """Test <s> in SSML for avoiding sentence break"""
        processor = TextProcessor(major_breaks={".", "!"})
        graph, root = processor("<s>First sentence. Second sentence!</s>", ssml=True)
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Sentences should not be split apart
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="First", text_with_ws="First "),
                Word(idx=1, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(
                    idx=2, sent_idx=0, text=".", text_with_ws=". ", is_major_break=True
                ),
                Word(idx=3, sent_idx=0, text="Second", text_with_ws="Second "),
                Word(idx=4, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(
                    idx=5, sent_idx=0, text="!", text_with_ws="!", is_major_break=True
                ),
            ],
        )

    def test_minor_breaks(self):
        """Test minor (phrase) break"""
        processor = TextProcessor(minor_breaks={","})
        graph, root = processor("this, is a test")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Comma should be split from word
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="this", text_with_ws="this"),
                Word(
                    idx=1, sent_idx=0, text=",", text_with_ws=", ", is_minor_break=True
                ),
                Word(idx=2, sent_idx=0, text="is", text_with_ws="is "),
                Word(idx=3, sent_idx=0, text="a", text_with_ws="a "),
                Word(idx=4, sent_idx=0, text="test", text_with_ws="test"),
            ],
        )

    def test_word_breaks(self):
        """Test inner-word break"""
        processor = TextProcessor(word_breaks={"-"})
        graph, root = processor("ninety-nine")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Word should be split
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ninety", text_with_ws="ninety "),
                Word(idx=1, sent_idx=0, text="nine", text_with_ws="nine"),
            ],
        )

    def test_spell_out(self):
        """Test interpret-as="spell-out" in SSML"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '<say-as interpret-as="spell-out">test123</say-as>', ssml=True
        )
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        print_graph(graph, root)

        # Word should be split into letters
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="t", text_with_ws="t "),
                Word(idx=1, sent_idx=0, text="e", text_with_ws="e "),
                Word(idx=2, sent_idx=0, text="s", text_with_ws="s "),
                Word(idx=3, sent_idx=0, text="t", text_with_ws="t "),
                Word(idx=4, sent_idx=0, text="one", text_with_ws="one "),
                Word(idx=5, sent_idx=0, text="two", text_with_ws="two "),
                Word(idx=6, sent_idx=0, text="three", text_with_ws="three"),
            ],
        )

    def test_initialisms(self):
        """Test initialism spell out"""
        processor = TextProcessor(
            major_breaks={"."},
            is_initialism=lambda s: s.isalpha() and s.isupper(),
            split_initialism=list,
        )
        graph, root = processor("TTS.")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Letters should be split
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="T", text_with_ws="T "),
                Word(idx=1, sent_idx=0, text="T", text_with_ws="T "),
                Word(idx=2, sent_idx=0, text="S", text_with_ws="S"),
                Word(
                    idx=3, sent_idx=0, text=".", text_with_ws=".", is_major_break=True
                ),
            ],
        )

    def test_numbers_one_language(self):
        """Test number verbalization (single language)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor("1 2 3")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Numbers should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(idx=1, sent_idx=0, text="two", text_with_ws="two "),
                Word(idx=2, sent_idx=0, text="three", text_with_ws="three"),
            ],
        )

    def test_numbers_multiple_languages(self):
        """Test number verbalization (SSML, multiple languages)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '1 <w lang="es_ES">2</w> <w lang="de_DE">3</w>', ssml=True
        )
        words = list(processor.words(graph, root, phonemes=False))

        # Numbers should be verbalized
        self.assertEqual(
            words,
            [
                Word(lang="en_US", idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(lang="es_ES", idx=1, sent_idx=0, text="dos", text_with_ws="dos "),
                Word(lang="de_DE", idx=2, sent_idx=0, text="drei", text_with_ws="drei"),
            ],
        )

    def test_currency_one_language(self):
        """Test currency verbalization (single language)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor("$10")
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Currency should be verbalized
        self.assertEqual(
            words,
            [
                Word(lang="en_US", idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(
                    lang="en_US",
                    idx=1,
                    sent_idx=0,
                    text="dollars",
                    text_with_ws="dollars",
                ),
            ],
        )

    def test_currency_multiple_language(self):
        """Test currency verbalization (SSML, multiple languages)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '€10 <w lang="fr_FR">€10</w> <w lang="nl_NL">€10</w>',
            ssml=True,
            phonemize=False,
        )
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Currencies should be verbalized
        self.assertEqual(
            words,
            [
                Word(lang="en_US", idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(
                    lang="en_US", idx=1, sent_idx=0, text="euro", text_with_ws="euro "
                ),
                Word(lang="fr_FR", idx=2, sent_idx=0, text="dix", text_with_ws="dix "),
                Word(
                    lang="fr_FR", idx=3, sent_idx=0, text="euros", text_with_ws="euros "
                ),
                Word(
                    lang="nl_NL", idx=4, sent_idx=0, text="tien", text_with_ws="tien "
                ),
                Word(lang="nl_NL", idx=5, sent_idx=0, text="euro", text_with_ws="euro"),
            ],
        )

    def test_currency_default(self):
        """Test default currency use when no currency symbol (interpret-as="currency")"""
        processor = TextProcessor(default_lang="en_US", default_currency="USD")
        graph, root = processor(
            '<say-as interpret-as="currency">10</say-as>', ssml=True
        )
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Currency should be verbalized, despite lack of "$" symbol
        self.assertEqual(
            words,
            [
                Word(lang="en_US", idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(
                    lang="en_US",
                    idx=1,
                    sent_idx=0,
                    text="dollars",
                    text_with_ws="dollars",
                ),
            ],
        )

    def test_time(self):
        """Test time verbalization (English)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor("  4:01pm")
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Time should be verbalized
        self.assertEqual(
            words,
            [
                Word(
                    lang="en_US", idx=0, sent_idx=0, text="four", text_with_ws="  four "
                ),
                Word(lang="en_US", idx=1, sent_idx=0, text="oh", text_with_ws="oh "),
                Word(lang="en_US", idx=2, sent_idx=0, text="one", text_with_ws="one "),
                Word(lang="en_US", idx=3, sent_idx=0, text="P", text_with_ws="P "),
                Word(lang="en_US", idx=4, sent_idx=0, text="M", text_with_ws="M"),
            ],
        )

    def test_time_no_colon(self):
        """Test time verbalization without a colon (English)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor("10am")
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Time should be verbalized
        self.assertEqual(
            words,
            [
                Word(lang="en_US", idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(lang="en_US", idx=1, sent_idx=0, text="A", text_with_ws="A "),
                Word(lang="en_US", idx=2, sent_idx=0, text="M", text_with_ws="M"),
            ],
        )

    def test_date_one_language(self):
        """Test date verbalization (single language)"""
        processor = TextProcessor(default_lang="en_US", word_breaks={"-"})
        graph, root = processor("4/1/1999")
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Date should be verbalized
        self.assertEqual(
            words,
            [
                Word(
                    lang="en_US", idx=0, sent_idx=0, text="April", text_with_ws="April "
                ),
                Word(
                    lang="en_US", idx=1, sent_idx=0, text="first", text_with_ws="first"
                ),
                Word(
                    lang="en_US",
                    idx=2,
                    sent_idx=0,
                    text=",",
                    text_with_ws=", ",
                    is_minor_break=True,
                ),
                Word(
                    lang="en_US",
                    idx=3,
                    sent_idx=0,
                    text="nineteen",
                    text_with_ws="nineteen ",
                ),
                Word(
                    lang="en_US",
                    idx=4,
                    sent_idx=0,
                    text="ninety",
                    text_with_ws="ninety ",
                ),
                Word(lang="en_US", idx=5, sent_idx=0, text="nine", text_with_ws="nine"),
            ],
        )

    def test_date_multiple_languages(self):
        """Test date verbalization (SSML, multiple languages)"""
        processor = TextProcessor(default_lang="en_US", word_breaks={"-"})
        graph, root = processor(
            '<speak><s>4/1/1999</s> <s lang="fr_FR">4/1/1999</s><s lang="de_DE">01.04.1999</s></speak>',
            ssml=True,
            phonemize=False,  # ensure French year is split
        )
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Date should be verbalized
        self.assertEqual(
            words,
            [
                # English
                Word(
                    lang="en_US", idx=0, sent_idx=0, text="April", text_with_ws="April "
                ),
                Word(
                    lang="en_US", idx=1, sent_idx=0, text="first", text_with_ws="first"
                ),
                Word(
                    lang="en_US",
                    idx=2,
                    sent_idx=0,
                    text=",",
                    text_with_ws=", ",
                    is_minor_break=True,
                ),
                Word(
                    lang="en_US",
                    idx=3,
                    sent_idx=0,
                    text="nineteen",
                    text_with_ws="nineteen ",
                ),
                Word(
                    lang="en_US",
                    idx=4,
                    sent_idx=0,
                    text="ninety",
                    text_with_ws="ninety ",
                ),
                Word(lang="en_US", idx=5, sent_idx=0, text="nine", text_with_ws="nine"),
                # French
                Word(
                    lang="fr_FR",
                    idx=0,
                    sent_idx=1,
                    text="quatrième",
                    text_with_ws="quatrième ",
                ),
                Word(
                    lang="fr_FR",
                    idx=1,
                    sent_idx=1,
                    text="janvier",
                    text_with_ws="janvier ",
                ),
                Word(
                    lang="fr_FR", idx=2, sent_idx=1, text="mille", text_with_ws="mille "
                ),
                Word(
                    lang="fr_FR", idx=3, sent_idx=1, text="neuf", text_with_ws="neuf "
                ),
                Word(
                    lang="fr_FR", idx=4, sent_idx=1, text="cent", text_with_ws="cent "
                ),
                Word(
                    lang="fr_FR",
                    idx=5,
                    sent_idx=1,
                    text="quatre",
                    text_with_ws="quatre ",
                ),
                Word(
                    lang="fr_FR", idx=6, sent_idx=1, text="vingt", text_with_ws="vingt "
                ),
                Word(lang="fr_FR", idx=7, sent_idx=1, text="dix", text_with_ws="dix "),
                Word(lang="fr_FR", idx=8, sent_idx=1, text="neuf", text_with_ws="neuf"),
                # German
                Word(
                    lang="de_DE",
                    idx=0,
                    sent_idx=2,
                    text="erste",
                    text_with_ws="erste ",
                ),
                Word(
                    lang="de_DE",
                    idx=1,
                    sent_idx=2,
                    text="April",
                    text_with_ws="April ",
                ),
                Word(
                    lang="de_DE",
                    idx=2,
                    sent_idx=2,
                    text="neunzehnhundertneunundneunzig",
                    text_with_ws="neunzehnhundertneunundneunzig",
                ),
            ],
        )

    def test_date_format_ordinal(self):
        """Test date format in SSML (ordinal)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '<say-as interpret-as="date" format="md">4/1</say-as>', ssml=True
        )
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Date is forced to be interpreted and format using day ordinal (first)
        self.assertEqual(
            words,
            [
                Word(
                    lang="en_US", idx=0, sent_idx=0, text="April", text_with_ws="April "
                ),
                Word(lang="en_US", idx=1, sent_idx=0, text="one", text_with_ws="one"),
            ],
        )

    def test_date_format_cardinal(self):
        """Test date format in SSML (cardinal)"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '<say-as interpret-as="date" format="dmy">4/1/2000</say-as>', ssml=True
        )
        words = list(processor.words(graph, root, phonemes=False, pos=False))

        # Date is forced to be interpreted and format using day ordinal (first)
        self.assertEqual(
            words,
            [
                Word(lang="en_US", idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(
                    lang="en_US", idx=1, sent_idx=0, text="April", text_with_ws="April "
                ),
                Word(lang="en_US", idx=2, sent_idx=0, text="two", text_with_ws="two "),
                Word(
                    lang="en_US",
                    idx=3,
                    sent_idx=0,
                    text="thousand",
                    text_with_ws="thousand",
                ),
            ],
        )

    def test_part_of_speech_tagging(self):
        """Test part-of-speech tagging"""

        def get_parts_of_speech(words, *args, **kwargs):
            return [w.upper() for w in words]

        processor = TextProcessor(
            # Made-up tagger that just gives the UPPER of the word back
            get_parts_of_speech=get_parts_of_speech
        )
        graph, root = processor("a test")
        words = list(processor.words(graph, root, explicit_lang=False, phonemes=False))

        # Fake POS tags are added
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="a", text_with_ws="a ", pos="A"),
                Word(idx=1, sent_idx=0, text="test", text_with_ws="test", pos="TEST"),
            ],
        )

    def test_phonemize_one_language(self):
        """Test phonemizer (single language)"""

        def lookup_phonemes(word: str, *args, **kwargs):
            return list(word)

        processor = TextProcessor(
            # Made-up phonemizer that just gives back the letters
            lookup_phonemes=lookup_phonemes,
        )
        graph, root = processor("test")
        words = list(processor.words(graph, root, pos=False, explicit_lang=False))

        # Single word is "phonemized"
        self.assertEqual(
            words,
            [
                Word(
                    idx=0,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["t", "e", "s", "t"],
                ),
            ],
        )

    def test_phonemize_one_language_multiple_roles(self):
        """Test phonemizer (SSML, multiple word roles)"""

        def lookup_phonemes(word, role=None, **kwargs):
            return list(word) if not role else list(word.upper())

        processor = TextProcessor(
            # Made-up phonemizer that gives back upper-case letters if a role is provided
            lookup_phonemes=lookup_phonemes
        )

        # Use made-up role
        graph, root = processor(
            '<speak>test <w role="some_role">test</w></speak>', ssml=True, pos=False
        )
        words = list(processor.words(graph, root, pos=False, explicit_lang=False))

        # Single word is phonemized two different manners depending on role
        self.assertEqual(
            words,
            [
                Word(
                    idx=0,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test ",
                    phonemes=["t", "e", "s", "t"],
                ),
                Word(
                    idx=1,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["T", "E", "S", "T"],
                ),
            ],
        )

    def test_phonemize_multiple_languages(self):
        """Test phonemizer (SSML, multiple languages)"""

        def en_lookup_phonemes(word: str, *args, **kwargs):
            return list(word)

        def de_lookup_phonemes(word: str, *args, **kwargs):
            return list(word.upper())

        processor = TextProcessor(
            default_lang="en_US",
            lookup_phonemes=en_lookup_phonemes,
            settings={
                "de_DE": TextProcessorSettings(
                    lang="de_DE", lookup_phonemes=de_lookup_phonemes
                )
            },
        )
        graph, root = processor(
            '<speak>test <w lang="de_DE">test</w></speak>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Single word is phonemized according to the lexicon with two different languages
        self.assertEqual(
            words,
            [
                Word(
                    lang="en_US",
                    idx=0,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test ",
                    phonemes=["t", "e", "s", "t"],
                ),
                Word(
                    lang="de_DE",
                    idx=1,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["T", "E", "S", "T"],
                ),
            ],
        )

    def test_sub(self):
        """Test SSML substitution"""
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '<speak><sub alias="World Wide Web Consortium">W3C</sub></speak>', ssml=True
        )
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Single word is replaced by multiple words
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="World", text_with_ws="World ",),
                Word(idx=1, sent_idx=0, text="Wide", text_with_ws="Wide ",),
                Word(idx=2, sent_idx=0, text="Web", text_with_ws="Web ",),
                Word(idx=3, sent_idx=0, text="Consortium", text_with_ws="Consortium",),
            ],
        )

    def test_break(self):
        """Test SSML break tag"""
        processor = TextProcessor(default_lang="en_US", keep_whitespace=False)
        graph, root = processor(
            """
        <speak>
          <break time="1s"/>
          <p>
            <break time="2s" />
            <s>
              <break time="3s" />
              Break <break time="4s" /> here
            </s>
            <break time="5s" />
          </p>
          <break time="6s" />
        </speak>
        """,
            ssml=True,
        )
        sentences = list(processor.sentences(graph, root, **WORDS_KWARGS))

        # Break times are attached to appropriate elements
        self.assertEqual(
            sentences,
            [
                Sentence(
                    idx=0,
                    text="Break here",
                    text_with_ws="Break here",
                    text_spoken="Break here",
                    pause_before_ms=((1 + 2) * 1000),
                    pause_after_ms=((5 + 6) * 1000),
                    words=[
                        Word(
                            idx=0,
                            sent_idx=0,
                            text="Break",
                            text_with_ws="Break",
                            pause_before_ms=(3 * 1000),
                            pause_after_ms=(4 * 1000),
                        ),
                        Word(idx=1, sent_idx=0, text="here", text_with_ws="here",),
                    ],
                ),
            ],
        )

    def test_mark(self):
        """Test SSML mark tag"""
        processor = TextProcessor(default_lang="en_US", keep_whitespace=False)
        graph, root = processor(
            """
        <speak>
          <mark name="a"/>
          <p>
            <mark name="b" />
            <s>
              <mark name="c" />
              Mark <mark name="d" /> here
            </s>
            <mark name="e" />
          </p>
          <mark name="f" />
        </speak>
        """,
            ssml=True,
        )
        sentences = list(processor.sentences(graph, root, **WORDS_KWARGS))

        # Mark names are attached to appropriate elements
        self.assertEqual(
            sentences,
            [
                Sentence(
                    idx=0,
                    text="Mark here",
                    text_with_ws="Mark here",
                    text_spoken="Mark here",
                    marks_before=["a", "b"],
                    marks_after=["e", "f"],
                    words=[
                        Word(
                            idx=0,
                            sent_idx=0,
                            text="Mark",
                            text_with_ws="Mark",
                            marks_before=["c"],
                            marks_after=["d"],
                        ),
                        Word(idx=1, sent_idx=0, text="here", text_with_ws="here",),
                    ],
                ),
            ],
        )

    def test_missing_speak(self):
        """Test SSML with missing <speak> tag"""
        processor = TextProcessor()
        graph, root = processor("<s>hello</s><s>world</s>", ssml=True,)
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # <speak> is automatically added when XML fails to parse
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="hello", text_with_ws="hello",),
                Word(idx=0, sent_idx=1, text="world", text_with_ws="world",),
            ],
        )

    def test_adjacent_voice(self):
        """Test SSML with adjacent <voice> tags"""
        processor = TextProcessor()
        graph, root = processor(
            '<voice name="a">hello.</voice><voice name="b">world.</voice>', ssml=True,
        )
        words = list(processor.words(graph, root, major_breaks=False, **WORDS_KWARGS))

        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, voice="a", text="hello", text_with_ws="hello",),
                Word(idx=0, sent_idx=1, voice="b", text="world", text_with_ws="world",),
            ],
        )

    def test_multiple_passes(self):
        """Test sentence that needs multiple passes to fully resolve"""
        processor = TextProcessor()
        graph, root = processor("ABCD-10")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # 1) ABCD-10 -> ABCD 10
        # 2) ABCD 10 -> A B C D ten
        self.assertEqual(
            words,
            [
                Word(idx=0, text="A", text_with_ws="A ",),
                Word(idx=1, text="B", text_with_ws="B ",),
                Word(idx=2, text="C", text_with_ws="C ",),
                Word(idx=3, text="D", text_with_ws="D ",),
                Word(idx=4, text="ten", text_with_ws="ten",),
            ],
        )

    def test_number_nonfinite(self):
        """Test sentence with nan or inf"""
        processor = TextProcessor()
        graph, root = processor("nan inf")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Words should not be parsed as numbers
        self.assertEqual(
            words,
            [
                Word(idx=0, text="nan", text_with_ws="nan ",),
                Word(idx=1, text="inf", text_with_ws="inf",),
            ],
        )

    def test_override_initialism(self):
        """Test use of inline lexicon pronunciation to override an initialism"""
        processor = TextProcessor()
        graph, root = processor("ROOFUS")
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Word is interpreted as initialism
        self.assertEqual(
            words,
            [
                Word(idx=0, text="R", text_with_ws="R ",),
                Word(idx=1, text="O", text_with_ws="O ",),
                Word(idx=2, text="O", text_with_ws="O ",),
                Word(idx=3, text="F", text_with_ws="F ",),
                Word(idx=4, text="U", text_with_ws="U ",),
                Word(idx=5, text="S", text_with_ws="S",),
            ],
        )

        graph, root = processor(
            """
        <speak>
          <lexicon>
            <lexeme>
              <grapheme>ROOFUS</grapheme>
              <phoneme>ɹ ˈu f ə s</phoneme>
            </lexeme>
          </lexicon>
          <s>ROOFUS</s>
        </speak>""",
            ssml=True,
        )
        words = list(processor.words(graph, root, **WORDS_KWARGS))

        # Word is *not* interpreted as initialism
        self.assertEqual(
            words, [Word(idx=0, text="ROOFUS", text_with_ws="ROOFUS",)],
        )


def print_graph_stderr(graph, root):
    """Print graph to stderr"""
    print_graph(graph, root, print_func=lambda *p: print(*p, file=sys.stderr))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
