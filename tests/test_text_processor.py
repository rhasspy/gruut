#!/usr/bin/env python3
import unittest

from gruut.text_processor import TextProcessor, Word, TextProcessorSettings, Sentence


class TextProcessorTestCase(unittest.TestCase):
    """Tests for TextProcessor"""

    def test_whitespace(self):
        processor = TextProcessor()
        graph, root = processor("This is  a   test    ")
        words = list(processor.words(graph, root))

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
        processor = TextProcessor(keep_whitespace=False)
        graph, root = processor("This is  a   test    ")
        words = list(processor.words(graph, root))

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

    def test_replacements(self):
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
        words = list(processor.words(graph, root))

        # Quotes and brackets are discarded
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="This", text_with_ws="This"),
                Word(idx=1, sent_idx=0, text=",", text_with_ws=", ", is_break=True),
                Word(idx=2, sent_idx=0, text="is", text_with_ws="is "),
                Word(idx=3, sent_idx=0, text="a", text_with_ws="a "),
                Word(idx=4, sent_idx=0, text="test", text_with_ws="test "),
                Word(idx=5, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(idx=6, sent_idx=0, text=".", text_with_ws=".", is_break=True),
            ],
        )

    def test_abbreviations(self):
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
        words = list(processor.words(graph, root))

        # Abbreviations are expanded, maintaining capitalization
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="Mister", text_with_ws="Mister"),
                Word(idx=1, sent_idx=0, text="?", text_with_ws="? ", is_break=True),
                Word(idx=0, sent_idx=1, text="I'm", text_with_ws="I'm "),
                Word(idx=1, sent_idx=1, text="just", text_with_ws="just "),
                Word(idx=2, sent_idx=1, text="a", text_with_ws="a "),
                Word(idx=3, sent_idx=1, text="doctor", text_with_ws="doctor"),
                Word(idx=4, sent_idx=1, text=",", text_with_ws=", ", is_break=True),
                Word(idx=5, sent_idx=1, text="on", text_with_ws="on "),
                Word(idx=6, sent_idx=1, text="this", text_with_ws="this "),
                Word(idx=7, sent_idx=1, text="Street", text_with_ws="Street "),
                Word(idx=8, sent_idx=1, text="at", text_with_ws="at "),
                Word(idx=9, sent_idx=1, text="least", text_with_ws="least"),
                Word(idx=10, sent_idx=1, text=".", text_with_ws=".", is_break=True),
            ],
        )

    def test_multiple_sentences(self):
        processor = TextProcessor(major_breaks={".", "!"})
        graph, root = processor("First  sentence. Second sentence! ")
        words = list(processor.words(graph, root))

        # Separated by a major break
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="First", text_with_ws="First  "),
                Word(idx=1, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(idx=2, sent_idx=0, text=".", text_with_ws=". ", is_break=True),
                Word(idx=0, sent_idx=1, text="Second", text_with_ws="Second "),
                Word(idx=1, sent_idx=1, text="sentence", text_with_ws="sentence"),
                Word(idx=2, sent_idx=1, text="!", text_with_ws="! ", is_break=True),
            ],
        )

        # Check sentences too
        sentences = list(processor.sentences(graph, root))
        self.assertEqual(
            sentences,
            [
                Sentence(
                    idx=0,
                    text="First sentence.",
                    text_with_ws="First  sentence. ",
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
                            is_break=True,
                        ),
                    ],
                ),
                Sentence(
                    idx=1,
                    text="Second sentence!",
                    text_with_ws="Second sentence! ",
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
                            is_break=True,
                        ),
                    ],
                ),
            ],
        )

    def test_explicit_sentence(self):
        processor = TextProcessor(major_breaks={".", "!"})
        graph, root = processor("<s>First sentence. Second sentence!</s>", ssml=True)
        words = list(processor.words(graph, root))

        # Sentences should not be split apart
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="First", text_with_ws="First "),
                Word(idx=1, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(idx=2, sent_idx=0, text=".", text_with_ws=". ", is_break=True),
                Word(idx=3, sent_idx=0, text="Second", text_with_ws="Second "),
                Word(idx=4, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(idx=5, sent_idx=0, text="!", text_with_ws="!", is_break=True),
            ],
        )

    def test_minor_breaks(self):
        processor = TextProcessor(minor_breaks={","})
        graph, root = processor("this, is a test")
        words = list(processor.words(graph, root))

        # Comma should be split from word
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="this", text_with_ws="this"),
                Word(idx=1, sent_idx=0, text=",", text_with_ws=", ", is_break=True),
                Word(idx=2, sent_idx=0, text="is", text_with_ws="is "),
                Word(idx=3, sent_idx=0, text="a", text_with_ws="a "),
                Word(idx=4, sent_idx=0, text="test", text_with_ws="test"),
            ],
        )

    def test_word_breaks(self):
        processor = TextProcessor(word_breaks={"-"})
        graph, root = processor("ninety-nine")
        words = list(processor.words(graph, root))

        # Word should be split
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ninety", text_with_ws="ninety "),
                Word(idx=1, sent_idx=0, text="nine", text_with_ws="nine "),
            ],
        )

    def test_spell_out(self):
        processor = TextProcessor()
        graph, root = processor(
            '<say-as interpret-as="spell-out">test</say-as>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Word should be split into letters
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="t", text_with_ws="t "),
                Word(idx=1, sent_idx=0, text="e", text_with_ws="e "),
                Word(idx=2, sent_idx=0, text="s", text_with_ws="s "),
                Word(idx=3, sent_idx=0, text="t", text_with_ws="t "),
            ],
        )

    def test_initialism(self):
        processor = TextProcessor(is_initialism=str.isupper, split_initialism=list)
        graph, root = processor("TTS")
        words = list(processor.words(graph, root))

        # Word should be split
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="T", text_with_ws="T "),
                Word(idx=1, sent_idx=0, text="T", text_with_ws="T "),
                Word(idx=2, sent_idx=0, text="S", text_with_ws="S "),
            ],
        )

    def test_numbers_one_language(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor("1 2 3")
        words = list(processor.words(graph, root))

        # Numbers should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(idx=1, sent_idx=0, text="two", text_with_ws="two "),
                Word(idx=2, sent_idx=0, text="three", text_with_ws="three "),
            ],
        )

    def test_numbers_multiple_languages(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '1 <w lang="es_ES">2</w> <w lang="de_DE">3</w>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Numbers should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(idx=1, sent_idx=0, text="dos", text_with_ws="dos "),
                Word(idx=2, sent_idx=0, text="drei", text_with_ws="drei "),
            ],
        )

    def test_currency_one_language(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor("$10")
        words = list(processor.words(graph, root))

        # Currency should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(idx=1, sent_idx=0, text="dollars", text_with_ws="dollars "),
            ],
        )

    def test_currency_multiple_language(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '€10 <w lang="fr_FR">€10</w> <w lang="nl_NL">€10</w>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Currencies should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(idx=1, sent_idx=0, text="euro", text_with_ws="euro "),
                Word(idx=2, sent_idx=0, text="dix", text_with_ws="dix "),
                Word(idx=3, sent_idx=0, text="euros", text_with_ws="euros "),
                Word(idx=4, sent_idx=0, text="tien", text_with_ws="tien "),
                Word(idx=5, sent_idx=0, text="euro", text_with_ws="euro "),
            ],
        )

    def test_currency_default(self):
        processor = TextProcessor(default_lang="en_US", default_currency="USD")
        graph, root = processor(
            '<say-as interpret-as="currency">10</say-as>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Currency should be verbalized, despite lack of "$" symbol
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(idx=1, sent_idx=0, text="dollars", text_with_ws="dollars "),
            ],
        )

    def test_date_one_language(self):
        processor = TextProcessor(default_lang="en_US", word_breaks={"-"})
        graph, root = processor("4/1/1999")
        words = list(processor.words(graph, root))

        # Date should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="April", text_with_ws="April "),
                Word(idx=1, sent_idx=0, text="first", text_with_ws="first "),
                Word(idx=2, sent_idx=0, text="nineteen", text_with_ws="nineteen "),
                Word(idx=3, sent_idx=0, text="ninety", text_with_ws="ninety "),
                Word(idx=4, sent_idx=0, text="nine", text_with_ws="nine "),
            ],
        )

    def test_date_multiple_languages(self):
        processor = TextProcessor(default_lang="en_US", word_breaks={"-"})
        graph, root = processor(
            '<speak><s>4/1/1999</s> <s lang="fr_FR">4/1/1999</s></speak>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Date should be verbalized
        self.assertEqual(
            words,
            [
                # English
                Word(idx=0, sent_idx=0, text="April", text_with_ws="April "),
                Word(idx=1, sent_idx=0, text="first", text_with_ws="first "),
                Word(idx=2, sent_idx=0, text="nineteen", text_with_ws="nineteen "),
                Word(idx=3, sent_idx=0, text="ninety", text_with_ws="ninety "),
                Word(idx=4, sent_idx=0, text="nine", text_with_ws="nine "),
                # French
                Word(idx=0, sent_idx=1, text="janvier", text_with_ws="janvier "),
                Word(idx=1, sent_idx=1, text="quatrième", text_with_ws="quatrième "),
                Word(idx=2, sent_idx=1, text="mille", text_with_ws="mille "),
                Word(idx=3, sent_idx=1, text="neuf", text_with_ws="neuf "),
                Word(idx=4, sent_idx=1, text="cent", text_with_ws="cent "),
                Word(idx=5, sent_idx=1, text="quatre", text_with_ws="quatre "),
                Word(idx=6, sent_idx=1, text="vingt", text_with_ws="vingt "),
                Word(idx=7, sent_idx=1, text="dix", text_with_ws="dix "),
                Word(idx=8, sent_idx=1, text="neuf", text_with_ws="neuf "),
            ],
        )

    def test_date_format_ordinal(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '<say-as interpret-as="date" format="md">4/1</say-as>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Date is forced to be interpreted and format using day ordinal (first)
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="April", text_with_ws="April "),
                Word(idx=1, sent_idx=0, text="one", text_with_ws="one "),
            ],
        )

    def test_date_format_cardinal(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor(
            '<say-as interpret-as="date" format="dmy">4/1/2000</say-as>', ssml=True
        )
        words = list(processor.words(graph, root))

        # Date is forced to be interpreted and format using day ordinal (first)
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(idx=1, sent_idx=0, text="April", text_with_ws="April "),
                Word(idx=2, sent_idx=0, text="two", text_with_ws="two "),
                Word(idx=3, sent_idx=0, text="thousand", text_with_ws="thousand "),
            ],
        )

    def test_part_of_speech_tagging(self):
        processor = TextProcessor(
            default_lang="en_US",
            # Made-up tagger that just gives the UPPER of the word back
            get_parts_of_speech=lambda words: [w.upper() for w in words],
        )
        graph, root = processor("a test")
        words = list(processor.words(graph, root))

        # Fake POS tags are added
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="a", text_with_ws="a ", pos="A"),
                Word(idx=1, sent_idx=0, text="test", text_with_ws="test", pos="TEST"),
            ],
        )

    def test_phonemize_one_language(self):
        processor = TextProcessor(
            default_lang="en_US",
            # Made-up phonemizer that just gives back the letters
            lookup_phonemes=lambda word, role: list(word),
        )
        graph, root = processor("test")
        words = list(processor.words(graph, root))

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
        processor = TextProcessor(
            default_lang="en_US",
            # Made-up phonemizer that gives back upper-case letters if a role is provided
            lookup_phonemes=lambda word, role: list(word)
            if not role
            else list(word.upper()),
        )

        # Use made-up role
        graph, root = processor(
            '<speak>test <w role="some_role">test</w></speak>', ssml=True
        )
        words = list(processor.words(graph, root))

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
        processor = TextProcessor(
            default_lang="en_US",
            lookup_phonemes=lambda word, role: list(word),
            settings={
                "de_DE": TextProcessorSettings(
                    lang="de_DE", lookup_phonemes=lambda word, role: list(word.upper())
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


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
