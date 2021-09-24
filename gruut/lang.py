#!/usr/bin/env python3
"""Language-specific settings"""
import logging
import re
import sqlite3
import typing
from pathlib import Path

import networkx as nx

from gruut.const import PHONEMES_TYPE, GraphType, SentenceNode
from gruut.g2p import GraphemesToPhonemes
from gruut.phonemize import SqlitePhonemizer
from gruut.pos import PartOfSpeechTagger
from gruut.text_processor import InterpretAsFormat, TextProcessorSettings
from gruut.utils import find_lang_dir, resolve_lang

_LOGGER = logging.getLogger("gruut")

# -----------------------------------------------------------------------------


def get_settings(
    lang: str,
    search_dirs: typing.Optional[typing.Iterable[typing.Union[str, Path]]] = None,
    lang_dir: typing.Optional[typing.Union[str, Path]] = None,
    model_prefix: str = "",
    load_pos_tagger: bool = True,
    load_phoneme_lexicon: bool = True,
    load_g2p_guesser: bool = True,
    **settings_args,
) -> TextProcessorSettings:
    """Get settings for a specific language"""
    # Resolve language
    if model_prefix:
        # espeak
        lang_model_prefix = model_prefix
        lang_only = lang
    elif "/" in lang:
        # en-us/espeak
        lang_only, lang_model_prefix = lang.split("/", maxsplit=1)
    else:
        # en-us
        lang_only = lang
        lang_model_prefix = ""

    # en_US -> en-us
    lang_only = resolve_lang(lang_only)

    if lang_model_prefix:
        lang_with_prefix = f"{lang_only}/{lang_model_prefix}"
    else:
        lang_with_prefix = lang_only

    if lang_dir is None:
        # Search for language data files
        lang_dir = find_lang_dir(lang_with_prefix, search_dirs=search_dirs)

    if lang_dir is not None:
        lang_dir = Path(lang_dir)

        # Part of speech tagger
        if load_pos_tagger:
            pos_model_path = lang_dir / "pos" / "model.crf"
            if pos_model_path.is_file():
                # POS tagger model will load on first use
                settings_args["get_parts_of_speech"] = DelayedPartOfSpeechTagger(
                    pos_model_path
                )
            else:
                _LOGGER.debug(
                    "(%s) no part of speech tagger found at %s", lang, pos_model_path,
                )

        # Phonemizer
        if load_phoneme_lexicon:
            lexicon_db_path = lang_dir / model_prefix / "lexicon.db"
            if lexicon_db_path.is_file():
                # Lower-case word if it can't be found in the lexicon
                phonemizer_args = {"word_transform_funcs": [str.lower]}

                settings_args["lookup_phonemes"] = DelayedSqlitePhonemizer(
                    lexicon_db_path, **phonemizer_args
                )
            else:
                _LOGGER.debug(
                    "(%s) no phoneme lexicon database found at %s",
                    lang,
                    lexicon_db_path,
                )

        # Grapheme to phoneme model
        if load_g2p_guesser:
            g2p_model_path = lang_dir / model_prefix / "g2p" / "model.crf"
            if g2p_model_path.is_file():
                settings_args["guess_phonemes"] = DelayedGraphemesToPhonemes(
                    g2p_model_path, transform_func=str.lower
                )

            else:
                _LOGGER.debug(
                    "(%s) no grapheme to phoneme CRF model found at %s",
                    lang,
                    g2p_model_path,
                )

    # ---------------------------------
    # Create language-specific settings
    # ---------------------------------

    if lang_only == "ar":
        # Arabic
        return get_ar_settings(lang_dir, **settings_args)

    if lang_only == "cs-cz":
        # Czech
        return get_cs_settings(lang_dir, **settings_args)

    if lang_only in {"en-us", "en-gb"}:
        # English
        return get_en_us_settings(lang_dir, **settings_args)

    if lang_only == "de-de":
        # German
        return get_de_settings(lang_dir, **settings_args)

    if lang_only in {"es-es", "es-mx"}:
        # Spanish
        return get_es_settings(lang_dir, **settings_args)

    if lang_only == "fa":
        # Farsi
        return get_fa_settings(lang_dir, **settings_args)

    if lang_only == "fr-fr":
        # French
        return get_fr_settings(lang_dir, **settings_args)

    if lang_only == "it-it":
        # Italian
        return get_it_settings(lang_dir, **settings_args)

    if lang_only == "nl":
        # Dutch
        return get_nl_settings(lang_dir, **settings_args)

    if lang_only == "pt":
        # Portuguese
        return get_pt_settings(lang_dir, **settings_args)

    if lang_only == "ru-ru":
        # Russian
        return get_ru_settings(lang_dir, **settings_args)

    if lang_only == "sv-se":
        # Swedish
        return get_sv_settings(lang_dir, **settings_args)

    if lang_only == "sw":
        # Swahili
        return get_sw_settings(lang_dir, **settings_args)

    # Default settings only
    return TextProcessorSettings(lang=lang, **settings_args)


# -----------------------------------------------------------------------------
# Arabic (ar, اَلْعَرَبِيَّةُ)
# -----------------------------------------------------------------------------


class ArabicPreProcessText:
    """Pre-processes text using mishkal"""

    def __call__(self, text: str) -> str:
        try:
            import mishkal.tashkeel

            # Load vocalizer
            if not hasattr(self, "vocalizer"):
                vocalizer = mishkal.tashkeel.TashkeelClass()
                setattr(self, "vocalizer", vocalizer)
            else:
                vocalizer = getattr(self, "vocalizer")

            assert vocalizer is not None

            # Add diacritics
            text = vocalizer.tashkeel(text)
        except ImportError:
            _LOGGER.warning("mishkal is highly recommended for language 'ar'")
            _LOGGER.warning("pip install 'mishkal>=0.4.0'")

        return text


def get_ar_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Arabic"""
    settings_args = {
        "major_breaks": {".", "؟", "!"},
        "minor_breaks": {"،", ";", ":"},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        "pre_process_text": ArabicPreProcessText(),
        **settings_args,
    }
    return TextProcessorSettings(lang="ar", **settings_args)


# -----------------------------------------------------------------------------
# Czech (cs-cz, čeština)
# -----------------------------------------------------------------------------


def get_cs_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Czech"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":"},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "’", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">", "’"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="cs_CZ", **settings_args)


# -----------------------------------------------------------------------------
# English (en-us, en-gb)
# -----------------------------------------------------------------------------


# TTS and T.T.S.
EN_INITIALISM_PATTERN = re.compile(r"^[A-Z]{2,}$")
EN_INITIALISM_DOTS_PATTERN = re.compile(r"^(?:[a-zA-Z]\.){2,}$")

EN_NON_WORD_PATTERN = re.compile(r"^(\W|_)+$")


def en_is_initialism(text: str) -> bool:
    """True if text is of the form TTS or T.T.S."""
    return (EN_INITIALISM_PATTERN.match(text) is not None) or (
        EN_INITIALISM_DOTS_PATTERN.match(text) is not None
    )


def get_en_us_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for English"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "*", "_"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">", "*", "_"},
        "default_currency": "USD",
        "default_date_format": InterpretAsFormat.DATE_MDY_ORDINAL,
        "is_initialism": en_is_initialism,
        "split_initialism": lambda text: list(text.replace(".", "")),
        "is_non_word": lambda text: EN_NON_WORD_PATTERN.match(text) is not None,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B'", '"'),  # replace single quotes (left)
            ("'\\B", '"'),  # replace signed quotes (right)
        ],
        "abbreviations": {
            r"^([cC])o\.": r"\1ompany",  # co. -> company
            r"^([dD])r\.": r"\1octor",  # dr. -> doctor
            r"^([dD])rs\.": r"\1octors",  # drs. -> doctors
            r"^([jJ])r\.": r"\1unior",  # jr. -> junior
            r"^([lL])td\.": r"\1imited",  # -> ltd. -> limited
            r"^([mM])r\.": r"\1ister",  # -> mr. -> mister
            r"^([mM])s\.": r"\1iss",  # -> ms. -> miss
            r"^([mM])rs\.": r"\1isess",  # -> mrs. -> misess
            r"^([sS])t\.": r"\1treet",  # -> st. -> street
            r"^([vV])s\.?": r"\1ersus",  # -> vs. -> versus
            r"(.*\d)%": r"\1 percent",  # % -> percent
            r"^&\s*$": "and",  # &-> and
        },
        "spell_out_words": {
            ".": "dot",
            "-": "dash",
            "@": "at",
            "*": "star",
            "+": "plus",
            "/": "slash",
        },
        **settings_args,
    }

    return TextProcessorSettings(lang="en_US", **settings_args)


# -----------------------------------------------------------------------------
# German (de-de)
# -----------------------------------------------------------------------------


def get_de_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for German"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "’", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">", "’"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="de_DE", **settings_args)


# -----------------------------------------------------------------------------
# Spanish (es-es, Español)
# -----------------------------------------------------------------------------


def get_es_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Spanish"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "¡", "¿"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="es_ES", **settings_args)


# -----------------------------------------------------------------------------
# Farsi/Persian (fa, فارسی)
# -----------------------------------------------------------------------------


class FarsiPartOfSpeechTagger:
    """Add POS tags with hazm"""

    def __init__(self, lang_dir: Path):
        self.lang_dir = lang_dir

    def __call__(self, words: typing.Sequence[str]) -> typing.Sequence[str]:
        pos_tags = []

        try:
            import hazm

            # Load normalizer
            if not hasattr(self, "normalizer"):
                normalizer = hazm.Normalizer()
                setattr(self, "normalizer", normalizer)

            # Load tagger
            if not hasattr(self, "tagger"):
                # Load part of speech tagger
                model_path = self.lang_dir / "pos" / "postagger.model"
                tagger = hazm.POSTagger(model=str(model_path))
                setattr(self, "tagger", tagger)

            text = " ".join(words)
            for sentence in hazm.sent_tokenize(normalizer.normalize(text)):
                for _word, pos in tagger.tag(hazm.word_tokenize(sentence)):
                    pos_tags.append(pos)
        except ImportError:
            _LOGGER.warning("hazm is highly recommended for language 'fa'")
            _LOGGER.warning("pip install 'hazm>=0.7.0'")

        return pos_tags


def fa_post_process_sentence(
    graph: GraphType, sent_node: SentenceNode, settings: TextProcessorSettings
):
    """Add e̞ for genitive case"""
    from gruut.text_processor import DATA_PROP, WordNode

    for dfs_node in nx.dfs_preorder_nodes(graph, sent_node.node):
        if not graph.out_degree(dfs_node) == 0:
            # Only leave
            continue

        node = graph.nodes[dfs_node][DATA_PROP]
        if isinstance(node, WordNode):
            word = typing.cast(WordNode, node)
            if word.phonemes and (word.pos == "Ne"):
                if isinstance(word.phonemes, list):
                    word.phonemes.append("e̞")
                else:
                    word.phonemes = list(word.phonemes) + ["e̞"]


def get_fa_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Farsi"""
    settings_args = {
        "major_breaks": {".", "؟", "!"},
        "minor_breaks": {",", ";", ":"},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "’", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">", "’"},
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        "post_process_sentence": fa_post_process_sentence,
        **settings_args,
    }

    if (lang_dir is not None) and ("get_parts_of_speech" not in settings_args):
        settings_args["get_parts_of_speech"] = FarsiPartOfSpeechTagger(lang_dir)

    return TextProcessorSettings(lang="fa", **settings_args)


# -----------------------------------------------------------------------------
# French (fr-fr, Français)
# -----------------------------------------------------------------------------


def fr_post_process_sentence(
    graph: GraphType, sent_node: SentenceNode, settings: TextProcessorSettings
):
    """Add liasons to phonemes"""
    from gruut.text_processor import DATA_PROP, WordNode
    from gruut.utils import sliding_window

    words = []
    for dfs_node in nx.dfs_preorder_nodes(graph, sent_node.node):
        if not graph.out_degree(dfs_node) == 0:
            # Only leave
            continue

        node = graph.nodes[dfs_node][DATA_PROP]
        if isinstance(node, WordNode):
            word_node = typing.cast(WordNode, node)
            words.append(word_node)

    for word1, word2 in sliding_window(words, 2):
        if word2 is None:
            continue

        if not (word1.text and word1.phonemes and word2.text and word2.phonemes):
            continue

        liason = False

        # Conditions to meet for liason check:
        # 1) word 1 ends with a silent consonant
        # 2) word 2 starts with a vowel (phoneme)

        last_char1 = word1.text[-1]
        ends_silent_consonant = fr_has_silent_consonant(last_char1, word1.phonemes[-1])
        starts_vowel = fr_is_vowel(word2.phonemes[0])

        if ends_silent_consonant and starts_vowel:
            # Handle mandatory liason cases
            # https://www.commeunefrancaise.com/blog/la-liaison

            if word1.text == "et":
                # No liason
                pass
            elif word1.pos in {"DET", "NUM"}:
                # Determiner/adjective -> noun
                liason = True
            elif (word1.pos == "PRON") and (word2.pos in {"AUX", "VERB"}):
                # Pronoun -> verb
                liason = True
            elif (word1.pos == "ADP") or (word1.text == "très"):
                # Preposition
                liason = True
            elif (word1.pos == "ADJ") and (word2.pos in {"NOUN", "PROPN"}):
                # Adjective -> noun
                liason = True
            elif word1.pos in {"AUX", "VERB"}:
                # Verb -> vowel
                liason = True

        if liason:
            # Apply liason
            # s -> z
            # p -> p
            # d|t -> d
            liason_pron = word1.phonemes

            if last_char1 in {"s", "x", "z"}:
                liason_pron.append("z")
            elif last_char1 == "d":
                liason_pron.append("t")
            elif last_char1 in {"t", "p", "n"}:
                # Final phoneme is same as char
                liason_pron.append(last_char1)


def fr_has_silent_consonant(last_char: str, last_phoneme: str) -> bool:
    """True if last consonant is silent in French"""
    # Credit: https://github.com/Remiphilius/PoemesProfonds/blob/master/lecture.py

    if last_char in {"d", "p", "t"}:
        return last_phoneme != last_char
    if last_char == "r":
        return last_phoneme != "ʁ"
    if last_char in {"s", "x", "z"}:
        return last_phoneme not in {"s", "z"}
    if last_char == "n":
        return last_phoneme not in {"n", "ŋ"}

    return False


def fr_is_vowel(phoneme: str) -> bool:
    """True if phoneme is a French vowel"""
    return phoneme in {
        "i",
        "y",
        "u",
        "e",
        "ø",
        "o",
        "ə",
        "ɛ",
        "œ",
        "ɔ",
        "a",
        "ɔ̃",
        "ɛ̃",
        "ɑ̃",
        "œ̃",
    }


def get_fr_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for French"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        "post_process_sentence": fr_post_process_sentence,
        **settings_args,
    }
    return TextProcessorSettings(lang="fr_FR", **settings_args)


# -----------------------------------------------------------------------------
# Italian (it-it, Italiano)
# -----------------------------------------------------------------------------


def get_it_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Italian"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        "post_process_sentence": fr_post_process_sentence,
        **settings_args,
    }
    return TextProcessorSettings(lang="it_IT", **settings_args)


# -----------------------------------------------------------------------------
# Dutch (nl, Nederlands)
# -----------------------------------------------------------------------------


def get_nl_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Dutch"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="nl", **settings_args)


# -----------------------------------------------------------------------------
# Portuguese (pt, Português)
# -----------------------------------------------------------------------------


def get_pt_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create default settings for Portuguese"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="pt", **settings_args)


# -----------------------------------------------------------------------------
# Russian (ru, Русский)
# -----------------------------------------------------------------------------


def get_ru_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Russian"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":"},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_currency": "RUB",
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="ru_RU", **settings_args)


# -----------------------------------------------------------------------------
# Swedish (sv-se, svenska)
# -----------------------------------------------------------------------------


def get_sv_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Swedish"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="sv_SE", **settings_args)


# -----------------------------------------------------------------------------
# Swahili (sw, Kiswahili)
# -----------------------------------------------------------------------------


def get_sw_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Swahili"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":"},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_date_format": InterpretAsFormat.DATE_DMY_ORDINAL,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("\\B['‘]", '"'),  # replace single quotes (left)
            ("['’]\\B", '"'),  # replace signed quotes (right)
        ],
        **settings_args,
    }
    return TextProcessorSettings(lang="sw", **settings_args)


# -----------------------------------------------------------------------------


class DelayedGraphemesToPhonemes:
    """Grapheme to phoneme guesser that loads on first use"""

    def __init__(
        self,
        model_path: typing.Union[str, Path],
        transform_func: typing.Optional[typing.Callable[[str], str]] = None,
        **g2p_args,
    ):
        self.model_path = model_path
        self.g2p: typing.Optional[GraphemesToPhonemes] = None
        self.transform_func = transform_func
        self.g2p_args = g2p_args

    def __call__(
        self, word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        if self.g2p is None:
            _LOGGER.debug(
                "Loading grapheme to phoneme CRF model from %s", self.model_path
            )
            self.g2p = GraphemesToPhonemes(self.model_path, **self.g2p_args)

        assert self.g2p is not None

        if self.transform_func is not None:
            word = self.transform_func(word)

        return self.g2p(word)


class DelayedPartOfSpeechTagger:
    """POS tagger that loads on first use"""

    def __init__(self, model_path: typing.Union[str, Path], **tagger_args):

        self.model_path = Path(model_path)
        self.tagger: typing.Optional[PartOfSpeechTagger] = None
        self.tagger_args = tagger_args

    def __call__(self, words: typing.Sequence[str]) -> typing.Sequence[str]:
        if self.tagger is None:
            _LOGGER.debug("Loading part of speech tagger from %s", self.model_path)
            self.tagger = PartOfSpeechTagger(self.model_path, **self.tagger_args)

        assert self.tagger is not None
        return self.tagger(words)


class DelayedSqlitePhonemizer:
    """Phonemizer that loads on first use"""

    def __init__(self, db_path: typing.Union[str, Path], **phonemizer_args):

        self.db_path = Path(db_path)
        self.phonemizer: typing.Optional[SqlitePhonemizer] = None
        self.phonemizer_args = phonemizer_args

    def __call__(
        self, word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        if self.phonemizer is None:
            _LOGGER.debug("Connecting to lexicon database at %s", self.db_path)
            db_conn = sqlite3.connect(self.db_path)
            self.phonemizer = SqlitePhonemizer(db_conn=db_conn, **self.phonemizer_args)

        assert self.phonemizer is not None
        return self.phonemizer(word, role=role)
