#!/usr/bin/env python3
"""Language-specific settings"""
import logging
import re
import sqlite3
import typing
from collections import deque
from pathlib import Path

import networkx as nx

from gruut.const import PHONEMES_TYPE, GraphType, SentenceNode, Time
from gruut.g2p import GraphemesToPhonemes
from gruut.phonemize import SqlitePhonemizer
from gruut.pos import PartOfSpeechTagger
from gruut.text_processor import (
    DATA_PROP,
    BreakNode,
    BreakWordNode,
    InterpretAsFormat,
    PunctuationWordNode,
    TextProcessorSettings,
    WordNode,
)
from gruut.utils import (
    find_lang_dir,
    remove_non_word_chars,
    resolve_lang,
    sliding_window,
)

_LOGGER = logging.getLogger("gruut.lang")

# -----------------------------------------------------------------------------


def get_settings(
    lang: str,
    search_dirs: typing.Optional[typing.Iterable[typing.Union[str, Path]]] = None,
    lang_dir: typing.Optional[typing.Union[str, Path]] = None,
    model_prefix: typing.Optional[str] = None,
    load_pos_tagger: bool = True,
    load_phoneme_lexicon: bool = True,
    load_g2p_guesser: bool = True,
    **settings_args,
) -> TextProcessorSettings:
    """Get settings for a specific language"""
    model_prefix = model_prefix or ""

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

    if lang_dir is None:
        # Search for language data files
        lang_dir = find_lang_dir(lang_only, search_dirs=search_dirs)

    if lang_dir is not None:
        lang_dir = Path(lang_dir)

        # Part of speech tagger
        if load_pos_tagger and ("get_parts_of_speech" not in settings_args):
            pos_model_path = lang_dir / "pos" / "model.crf"
            if pos_model_path.is_file():
                # POS tagger model will load on first use
                settings_args["get_parts_of_speech"] = DelayedPartOfSpeechTagger(
                    pos_model_path
                )
            else:
                _LOGGER.debug(
                    "(%s) no part of speech tagger found at %s",
                    lang,
                    pos_model_path,
                )

        # Phonemizer
        if load_phoneme_lexicon and ("lookup_phonemes" not in settings_args):
            lexicon_db_path = lang_dir / lang_model_prefix / "lexicon.db"
            if lexicon_db_path.is_file():
                # Transformations to apply to words when they can't be found in the lexicon
                phonemizer_args = {
                    "word_transform_funcs": [
                        str.lower,
                        remove_non_word_chars,
                        lambda s: remove_non_word_chars(s.lower()),
                    ],
                }

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
        if load_g2p_guesser and ("guess_phonemes" not in settings_args):
            g2p_model_path = lang_dir / lang_model_prefix / "g2p" / "model.crf"
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

    if lang_only in {"ca-ce", "ca-ba", "ca-no", "ca-va"}:
        # Catalan
        return get_ca_settings(lang_dir, **settings_args)

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

    if lang_only == "lb":
        # Lëtzebuergesch
        return get_lb_settings(lang_dir, **settings_args)

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

    if lang_only == "zh-cn":
        # Chinese
        return get_zh_settings(lang_dir, **settings_args)

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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
        **settings_args,
    }
    return TextProcessorSettings(lang="cs_CZ", **settings_args)


# -----------------------------------------------------------------------------
# English (en-us, en-gb)
# -----------------------------------------------------------------------------


# TTS and T.T.S.
EN_INITIALISM_PATTERN = re.compile(r"^\s*[A-Z]{2,}\s*$")
EN_INITIALISM_DOTS_PATTERN = re.compile(r"^(?:\s*[a-zA-Z]\.){1,}\s*$")

EN_NON_WORD_PATTERN = re.compile(r"^(\W|_)+$")
EN_ORDINAL_PATTERN = re.compile(r"^(-?[0-9][0-9,]*)(?:st|nd|rd|th).*$")

EN_TIME_PATTERN = re.compile(
    r"""^((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
         (?::
         ([0-5][0-9]))?                          # minutes
         \s*(a\.m\.|am|pm|p\.m\.|a\.m|p\.m)? # am/pm
         $""",
    re.IGNORECASE | re.X,
)

EN_MAYBE_DATE_PATTERN = re.compile(r"[0-9]+[-/][0-9]+")
EN_MAYBE_TIME_PATTERN = re.compile(r"[0-9]+[:ap]", re.IGNORECASE)


def en_is_initialism(text: str) -> bool:
    """True if text is of the form TTS or T.T.S."""
    return (EN_INITIALISM_PATTERN.match(text) is not None) or (
        EN_INITIALISM_DOTS_PATTERN.match(text) is not None
    )


def en_get_ordinal(text: str) -> typing.Optional[int]:
    """Parse English ordinal string (e.g., 1st -> 1)"""
    match = EN_ORDINAL_PATTERN.match(text)
    if match is not None:
        return int(re.sub(r"[^0-9]", "", match.group(1)))

    return None


def en_parse_time(text: str) -> typing.Optional[Time]:
    """Parse English clock time (e.g. 4:01pm)"""
    match = EN_TIME_PATTERN.match(text.strip().lower())
    if match is None:
        return None

    hours = int(match.group(1))
    maybe_minutes = match.group(6)
    minutes = 0 if maybe_minutes is None else int(maybe_minutes)
    period = match.group(7)

    if period is not None:
        # Normalize period
        if "a" in period:
            period = "A.M."
        else:
            period = "P.M."
    else:
        if ":" not in text:
            # Require a colon if no period is specified to avoid parsing plain
            # numbers like "1" into time expressions.
            return None

    return Time(hours=hours, minutes=minutes, period=period)


def en_verbalize_time(time: Time) -> typing.Iterable[str]:
    """Convert time into words"""

    hour = time.hours

    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12

    yield str(hour)

    minute = time.minutes
    if minute > 0:
        if minute < 10:
            yield "oh"

        yield str(minute)

    if time.period is not None:
        yield time.period


def en_is_maybe_date(s: str) -> bool:
    """True if string is maybe a U.S. English date"""
    return EN_MAYBE_DATE_PATTERN.match(s) is not None


def en_is_maybe_time(s: str) -> bool:
    """True if string is maybe a U.S. English time"""
    return EN_MAYBE_TIME_PATTERN.match(s) is not None


def get_en_us_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for English"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "'", "“", "«", "[", "(", "<", "*", "_"},
        "end_punctuations": {'"', "'", "”", "»", "]", ")", ">", "*", "_"},
        "default_currency": "USD",
        "default_date_format": "{m} {o}, {y}",
        "is_initialism": en_is_initialism,
        "split_initialism": lambda text: list(text.replace(".", "")),
        "is_non_word": lambda text: EN_NON_WORD_PATTERN.match(text) is not None,
        "get_ordinal": en_get_ordinal,
        "parse_time": en_parse_time,
        "verbalize_time": en_verbalize_time,
        "replacements": [("’", "'")],  # normalize apostrophe
        "abbreviations": {
            r"^([cC])o\.": r"\1ompany",  # co. -> company
            r"^([dD])r\.": r"\1octor",  # dr. -> doctor
            r"^([dD])rs\.": r"\1octors",  # drs. -> doctors
            r"^([jJ])r\.('s)?": r"\1unior\2",  # jr. -> junior
            r"^([lL])td\.": r"\1imited",  # -> ltd. -> limited
            r"^([mM])r\.": r"\1ister",  # -> mr. -> mister
            r"^([mM])s\.": r"\1iss",  # -> ms. -> miss
            r"^([mM])rs\.": r"\1issus",  # -> mrs. -> missus
            r"^([sS])t\.": r"\1treet",  # -> st. -> street
            r"^([vV])s\.?": r"\1ersus",  # -> vs. -> versus
            r"(.*\d)%": r"\1 percent",  # % -> percent
            r"^&(\s*)$": r"and\1",  # &-> and
            r"^([mM])t\.": r"\1ount",  # -> mt. -> mount
            # Roman numerals
            "^II$": "two",
            "^III$": "three",
            "^IV$": "four",
            "^VI$": "six",
            "^VII$": "seven",
            "^VIII$": "eight",
        },
        "spell_out_words": {
            ".": "dot",
            "-": "dash",
            "@": "at",
            "*": "star",
            "+": "plus",
            "/": "slash",
        },
        "is_maybe_time": en_is_maybe_time,
        "is_maybe_date": en_is_maybe_date,
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
            ("ß", "ss"),  # normalize Eszett
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
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
            normalizer = getattr(self, "normalizer", None)
            if normalizer is None:
                normalizer = hazm.Normalizer()
                setattr(self, "normalizer", normalizer)

            # Load tagger
            tagger = getattr(self, "tagger", None)
            if tagger is None:
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
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
        "replacements": [("’", "'")],  # normalize apostrophe
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
        "post_process_sentence": fr_post_process_sentence,
        **settings_args,
    }
    return TextProcessorSettings(lang="it_IT", **settings_args)


# -----------------------------------------------------------------------------
# Luxembourgish (lb, Lëtzebuergesch)
# -----------------------------------------------------------------------------


def get_lb_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Luxembourgish"""
    settings_args = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"-", "_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
        "babel_locale": "lb",
        **settings_args,
    }
    return TextProcessorSettings(lang="lb", **settings_args)


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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
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
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [("’", "'")],  # normalize apostrophe
        **settings_args,
    }
    return TextProcessorSettings(lang="sw", **settings_args)


# -----------------------------------------------------------------------------
# Chinese (zh-cn, 汉语)
# -----------------------------------------------------------------------------


def get_zh_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    """Create settings for Chinese"""

    # https://en.wikipedia.org/wiki/Chinese_punctuation
    settings_args = {
        "major_breaks": {"。", "！", "？"},
        "minor_breaks": {"；", "：", "，", "、", "……"},
        "begin_punctuations": {"（", "［", "【", "「", "﹁", '"', "《", "〈"},
        "end_punctuations": {"）", "］", " 】", "」", "﹂", '"', "》", "〉"},
        "word_breaks": {"‧"},
        "split_words": list,
        "join_str": "",
        **settings_args,
    }
    return TextProcessorSettings(lang="zh_CN", **settings_args)


# -----------------------------------------------------------------------------
# Catalan (ca, Catalan)
# -----------------------------------------------------------------------------

# Pre-Process constants
# Same for all accents in this version
VOWEL_CHARS = [
    "a",
    "ä",
    "à",
    "e",
    "ë",
    "é",
    "è",
    "i",
    "í",
    "ï",
    "o",
    "ö",
    "ó",
    "ò",
    "u",
    "ü",
    "ú",
]
ACCENTED_VOWEL_CHARS = ["à", "é", "è", "í", "ó", "ò", "ú"]
NUCLITIC_CHARS = ["a", "à", "e", "é", "è", "í", "ï", "o", "ó", "ò", "ú"]
ACCENT_CHANGES = {
    "a": "à",
    "e": "é",
    "i": "í",
    "ï": "í",
    "o": "ó",
    "u": "ú",
    "ü": "ú",
}
INSEPARABLES = [
    "bh",
    "bl",
    "br",
    "ch",
    "cl",
    "cr",
    "dh",
    "dj",
    "dr",
    "fh",
    "fh",
    "fl",
    "fr",
    "gh",
    "gl",
    "gr",
    "gu",
    "gü",
    "jh",
    "kh",
    "kl",
    "kr",
    "lh",
    "ll",
    "mh",
    "nh",
    "ny",
    "ph",
    "pl",
    "pr",
    "qu",
    "qü",
    "rh",
    "sh",
    "th",
    "th",
    "tr",
    "vh",
    "wh",
    "xh",
    "xh",
    "yh",
    "zh",
]
VOC_IR = ["cuir", "vair"]
EINESGRAM = [
    "-de-",
    "-en",
    "-hi",
    "-ho",
    "-i",
    "-i-",
    "-la",
    "-les",
    "-li",
    "-lo",
    "-los",
    "-me",
    "-ne",
    "-nos",
    "-se",
    "-te",
    "-us",
    "-vos",
    "a",
    "a-",
    "al",
    "als",
    "amb",
    "bi-",
    "co",
    "de",
    "de-",
    "del",
    "dels",
    "el",
    "els",
    "em",
    "en",
    "ens",
    "es",
    "et",
    "hi",
    "ho",
    "i",
    "i-",
    "la",
    "les",
    "li",
    "lo",
    "ma",
    "me",
    "mon",
    "na",
    "pel",
    "pels",
    "per",
    "que",
    "re",
    "sa",
    "se",
    "ses",
    "si",
    "sos",
    "sub",
    "ta",
    "te",
    "tes",
    "ton",
    "un",
    "uns",
    "us",
]
EXCEP_ACC = {
    "antropologico": "antropològico",
    "arterio": "artèrio",
    "artistico": "artístico",
    "basquet": "bàsquet",
    "cardio": "càrdio",
    "catolico": "catòlico",
    "cientifico": "científico",
    "circum": "círcum",
    "civico": "cívico",
    "democrata": "demòcrata",
    "democratico": "democràtico",
    "dumping": "dúmping",
    "economico": "econòmico",
    "edgar": "èdgar",
    "fenicio": "fenício",
    "filosofico": "filosòfico",
    "fisico": "físico",
    "fisio": "físio",
    "geografico": "geogràfico",
    "hetero": "hétero",
    "higenico": "higènico",
    "higienico": "higiènico",
    "hiper": "híper",
    "historico": "històrico",
    "ibero": "íbero",
    "ideologico": "ideològico",
    "input": "ínput",
    "inter": "ínter",
    "jonatan": "jònatan",
    "juridico": "jurídico",
    "labio": "làbio",
    "linguo": "línguo",
    "literario": "literàrio",
    "logico": "lògico",
    "magico": "màgico",
    "maniaco": "maníaco",
    "marketing": "màrketing",
    "oxido": "òxido",
    "petroleo": "petròleo",
    "politico": "político",
    "quantum": "quàntum",
    "quimico": "químico",
    "quimio": "químio",
    "radio": "ràdio",
    "romanico": "romànico",
    "simbolico": "simbòlico",
    "socio": "sòcio",
    "super": "súper",
    "tecnico": "tècnico",
    "teorico": "teòrico",
    "tragico": "tràgico",
    "traqueo": "tràqueo",
}
DIFT_DECR = ["au", "ai", "eu", "ei", "ou", "oi", "iu", "àu", "ui"]
VOC_SOLA = ["a", "e", "i", "o", "u", "ï", "ü"]
VOC_MES_S = ["as", "es", "is", "os", "us", "às", "ès"]
EN_IN = ["en", "in", "àn"]

# Pre-Process functions and classes


# TODO review all functions, may need refactor
# TODO define depending the dialect
def vocal(carac: str) -> bool:

    return carac in VOWEL_CHARS


def acaba_en_vocal(prefix: str) -> bool:
    darrer = prefix[-1]
    return vocal(darrer)


def post_prefix_ok(resta: str) -> bool:

    mida = len(resta)
    primer = resta[0]
    segon = "\0"
    if mida > 1:
        segon = resta[1]

    if primer in "iu":
        return True
    elif primer in "rs":
        if mida > 1 and vocal(segon):
            return True
    return False


def nuclitica(carac: str) -> bool:
    return carac in NUCLITIC_CHARS


def gicf_suf(mot: str, pos: int, mots_voc_ir: typing.List[str]) -> bool:

    if mot[pos:].endswith("isme") and len(mot) - pos == 4:
        return True
    elif mot[pos:].endswith("ista") and len(mot) - pos == 4:
        return True
    elif mot[pos:].endswith("ismes") and len(mot) - pos == 5:
        return True
    elif mot[pos:].endswith("istes") and len(mot) - pos == 5:
        return True

    i1 = mot.find("ir")
    if i1 == pos and len(mot) - pos == 2:
        if mot in mots_voc_ir:
            return False
        else:
            return True

    i1 = mot.find("int")
    if i1 == pos and len(mot) - pos == 3:
        return True

    i1 = mot.find("iré")
    if i1 == pos and len(mot) - pos == 3:
        return True

    i1 = mot.find("iràs")
    if i1 == pos and len(mot) - pos == 4:
        return True

    i1 = mot.find("irà")
    if i1 == pos and len(mot) - pos == 3:
        return True

    i1 = mot.find("irem")
    if i1 == pos and len(mot) - pos == 4:
        return True

    i1 = mot.find("ireu")
    if i1 == pos and len(mot) - pos == 4:
        return True

    i1 = mot.find("iran")
    if i1 == pos and len(mot) - pos == 4:
        return True

    i1 = mot.find("iria")
    if i1 == pos and len(mot) - pos == 4:
        return True

    i1 = mot.find("iries")
    if i1 == pos and len(mot) - pos == 5:
        return True

    i1 = mot.find("iríem")
    if i1 == pos and len(mot) - pos == 5:
        return True

    i1 = mot.find("iríeu")
    if i1 == pos and len(mot) - pos == 5:
        return True

    i1 = mot.find("irien")
    if i1 == pos and len(mot) - pos == 5:
        return True

    return False


class Sillaba:
    def __init__(self, sil: str):

        self.text_ = sil
        self.tonica_ = False
        self.grafnuc_ = -1
        self.fonnuc_ = -1
        self.fons_ = deque()

    def grafnuc(self, nuc: int):
        self.grafnuc_ = nuc

    def get_grafnuc(self) -> int:
        return self.grafnuc_

    def get_text(self) -> str:
        return self.text_

    def get_text_at_index(self, idx: int) -> str:
        return self.text_[idx]

    def sizetext(self) -> int:
        return len(self.text_)

    def tonica(self) -> bool:
        self.tonica_ = True

    def asktonica(self) -> bool:
        return self.tonica_

    def es_sil_tonica(self) -> bool:

        if self.tonica_:
            return "sí"
        else:
            return "no"

    def numfons(self) -> int:
        return len(self.fons_)

    def allofon(self, fonidx: int) -> str:
        return self.fons_[fonidx]

    def allofons(self) -> deque:
        return self.fons_

    def push_back(self, fon: str):
        self.fons_.append(fon)

    def push_front(self, fon: str):
        self.fons_.insert(0, fon)

    def pop_front(self):
        self.fons_.popleft()

    def pop_back(self):
        self.fons_.pop()

    def empty(self) -> bool:
        return len(self.fons_) == 0

    def fonnuc(self, fnuc: int):
        self.fonnuc_ = fnuc

    def get_fonnuc(self) -> int:
        return self.fonnuc_


class Part:
    def __init__(self, tros: str):
        self.text_ = tros
        self.transsil_ = (
            deque()
        )  # It will be a deque structure with Sillaba instances as elements

    def push_back(self, sil: Sillaba):
        self.transsil_.append(sil)

    def pop_back(self):
        self.transsil_.pop()

    def pop_front(self):
        self.transsil_.popleft()

    def empty(self) -> bool:
        return len(self.transsil_) == 0

    def size(self) -> int:
        return len(self.transsil_)

    def tonica(self, silidx: int) -> bool:
        # self.transsil_[silidx] is an Sillaba instance, which has the attribute tonica_
        return self.transsil_[silidx].tonica_

    def idxgrafnucli(self, silidx: int) -> int:
        # self.transsil_[silidx] is an Sillaba instance, which has the attribute grafnuc_
        return self.transsil_[silidx].grafnuc_

    def grafnucli(self, silidx: int) -> str:
        # self.transsil_[silidx] is an Sillaba instance, which has an attributes text_ and grafnuc_
        return self.transsil_[silidx].text_[self.transsil_[silidx].grafnuc_]

    def sil(self, silnum: int) -> Sillaba:
        return self.transsil_[silnum]

    def sils(self) -> deque:
        return self.transsil_

    def text(self) -> str:
        return self.text_

    def textinici(self, silindex: int, charindex: int) -> str:

        # Gives the text of the previous syllables, and from the syllable silindex to charindex not included

        mot = ""
        for i in range(silindex):
            mot += self.transsil_[i].text_
        if charindex:
            mot += self.transsil_[silindex].text_[:charindex]
        return mot

    def textfinal(self, silindex: int, charindex: int) -> str:

        # Gives the text starting from the syllable silindex and the character charindex (included) and up to the end of the word

        mot = self.transsil_[silindex].text_[charindex:]
        for i in range(silindex + 1, len(self.transsil_)):
            mot += self.transsil_[i].text_
        return mot

    def textsilini(self, silindex: int, charindex: int) -> str:

        # gives the text of the syllable silindex, from the beginning to the character charindex not included
        return self.transsil_[silindex].text_[:charindex]

    def textsilfinal(self, silindex: int, charindex: int) -> str:

        # Gives the text of the syllable silindex, from charindex inclusive to the end
        return self.transsil_[silindex].text_[charindex:]

    def charidxsilini(self, silindex: int) -> int:

        car = self.transsil_[silindex].text_[0]
        if car == "'" or car == "-":
            return 1
        else:
            return 0

    def charidxsilfinal(self, silindex: int) -> int:

        siltxt = self.transsil_[silindex].text_
        car = siltxt[-1]
        if car == "'" or car == "-":
            return len(siltxt) - 2
        else:
            return len(siltxt) - 1


class MotNuclis:
    def __init__(self, mot: str, es_adverbi: bool):

        self.adverbi_ = es_adverbi
        self.el_mot = mot
        self.pos_nuclis = []

        self.load_insep()

    def load_insep(self):

        # Set self.insep_ and self.mots_voc_ir_

        self.insep_ = INSEPARABLES
        self.mots_voc_ir_ = VOC_IR

    def troba_nuclis_mot(self):

        mida = len(self.el_mot)
        adjectiu = ""

        if self.adverbi_:
            adjectiu = self.el_mot[0 : mida - 4]
            self.el_mot = adjectiu
            mida = len(self.el_mot)

        gr = 0
        while gr < mida:

            car = self.el_mot[gr]

            if nuclitica(car):
                self.pos_nuclis.append(gr)
                gr = gr + 1
                continue

            elif car == "i":
                # gicf o sufix
                if gicf_suf(self.el_mot, gr, self.mots_voc_ir_):
                    self.pos_nuclis.append(gr)
                    gr = gr + 1
                    continue
                else:
                    abans = self.el_mot[0:gr]
                    premida = len(abans)

                    if (premida == 0) or (premida == 1 and abans == "h"):
                        # casos iode o hiena, i, hi
                        if gr == mida - 1:
                            # i, hi
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue
                        elif vocal(self.el_mot[gr + 1]):
                            # hiena iode
                            gr = gr + 1
                            continue
                        # hissar, ira
                        else:
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue

                    elif (premida == 1) and (abans == "u"):

                        if gr == mida - 1 or self.el_mot[gr + 1] == "x":
                            gr = gr + 1
                            continue
                        else:
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue

                    elif (premida == 2) and (abans == "hu"):

                        if gr == mida - 1:
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue

                        if self.el_mot[gr + 1] == "x":
                            gr = gr + 1
                            continue
                        else:
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue

                    elif self.el_mot[gr - 1] == "u":
                        # tres vocals seguides vocal+u+i, la u es consonant i la "i" es nucli
                        if (premida > 1) and vocal(self.el_mot[gr - 2]):
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue
                        elif (premida > 1) and (
                            self.el_mot[gr - 2] == "q" or self.el_mot[gr - 2] == "g"
                        ):
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue
                        else:
                            # ui tot sol
                            gr = gr + 1
                            continue

                    elif self.el_mot[gr - 1] == "ü":

                        if (premida > 1) and (
                            self.el_mot[gr - 2] == "q" or self.el_mot[gr - 2] == "g"
                        ):
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue
                        else:
                            # üi no precedit de g,q
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue

                    elif vocal(self.el_mot[gr - 1]):
                        # vocal + i, la i no es nucli
                        gr = gr + 1
                        continue

                    else:
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue

            elif car == "u":

                abans = self.el_mot[0:gr]
                premida = len(abans)

                if (premida == 0) or (premida == 1 and abans == "h"):
                    # casos uadi o hu+vocal, u, hu
                    if gr == mida - 1:
                        # u, hu
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue
                    elif self.el_mot == "ui" or self.el_mot == "uix":
                        # potser se n'han d'afegir mes
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue
                    elif (pos := self.el_mot.find("ix")) != -1 and pos == gr + 1:
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue
                    elif vocal(self.el_mot[gr + 1]):
                        # uadi hu+vocal
                        gr = gr + 1
                        continue
                    else:
                        # huns, una
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue

                elif (premida == 1) and (abans == "i"):
                    self.pos_nuclis.append(gr)
                    gr = gr + 1
                    continue

                elif self.el_mot[gr - 1] == "i":
                    # tres vocals seguides vocal+i+u, la i es consonant i la "u" es nucli
                    if premida > 2:
                        boci = self.el_mot[gr - 3 : gr - 1]

                        if boci == "gu" or boci == "qu":
                            gr = gr + 1
                            continue

                        elif vocal(self.el_mot[gr - 2]):
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue

                        else:
                            gr = gr + 1
                            continue

                    elif premida == 2:
                        if vocal(self.el_mot[gr - 2]):
                            self.pos_nuclis.append(gr)
                            gr = gr + 1
                            continue
                        else:
                            gr = gr + 1
                            continue
                    else:
                        gr = gr + 1
                        continue

                elif self.el_mot[gr - 1] == "g" or self.el_mot[gr - 1] == "q":
                    if gr == mida - 1:
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue

                    elif vocal(self.el_mot[gr + 1]):
                        gr = gr + 1
                        continue

                    else:
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue

                elif self.el_mot[gr - 1] == "ü":
                    if (premida > 1) and (
                        self.el_mot[gr - 2] == "q" or self.el_mot[gr - 2] == "g"
                    ):
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue
                    else:
                        # üu no precedit de g,q
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue

                elif vocal(self.el_mot[gr - 1]):
                    # vocal + u, la u no es nucli
                    gr = gr + 1
                    continue

                else:
                    # tancara l'else de quan no es sufix
                    self.pos_nuclis.append(gr)
                    gr = gr + 1
                    continue

            elif car == "ü":

                pos = 0

                if (pos := self.el_mot.find("argü")) != -1:
                    if pos + 3 == gr:
                        self.pos_nuclis.append(gr)
                        self.pos_nuclis.append(gr + 1)
                        gr += 1
                        gr = gr + 1
                        continue
                    else:
                        gr = gr + 1
                        continue
                elif gr > 0:
                    if self.el_mot[gr - 1] == "g" or self.el_mot[gr - 1] == "q":
                        gr = gr + 1
                        continue
                    else:
                        self.pos_nuclis.append(gr)
                        gr = gr + 1
                        continue

            else:
                gr = gr + 1
                continue

        if self.adverbi_:
            self.el_mot += "ment"
            mida = len(self.el_mot)
            self.pos_nuclis.append(mida - 3)

    def inseparable(self, tros: str) -> bool:
        return tros in self.insep_

    def separa_sillabes(
        self, vec_sil: typing.List[str], els_nuclis: typing.List[int]
    ) -> typing.Tuple[typing.List[str], typing.List[int]]:

        fronteres = []

        if len(self.pos_nuclis) == 1:

            vec_sil.append(self.el_mot)
            els_nuclis.append(self.pos_nuclis[0])

            return vec_sil, els_nuclis

        # Set the fronteres vector
        for i in range(len(self.pos_nuclis) - 1):

            longi = self.pos_nuclis[i + 1] - self.pos_nuclis[i] - 1
            tros = self.el_mot[self.pos_nuclis[i] + 1 : self.pos_nuclis[i] + 1 + longi]

            # vocals contigues
            if longi == 0:
                fronteres.append(self.pos_nuclis[i])

            elif longi == 1:
                fronteres.append(self.pos_nuclis[i])

            elif longi == 2:
                if self.inseparable(
                    self.el_mot[self.pos_nuclis[i] + 1 : self.pos_nuclis[i] + 1 + 2]
                ):
                    fronteres.append(self.pos_nuclis[i])
                elif self.el_mot[self.pos_nuclis[i] + 2] == "h":
                    fronteres.append(self.pos_nuclis[i])
                else:
                    fronteres.append(self.pos_nuclis[i] + 1)

            elif longi == 3:
                if self.inseparable(
                    self.el_mot[self.pos_nuclis[i] + 2 : self.pos_nuclis[i] + 2 + 2]
                ):
                    if self.el_mot[self.pos_nuclis[i] + 1] == "-":
                        fronteres.append(self.pos_nuclis[i])
                    else:
                        fronteres.append(self.pos_nuclis[i] + 1)
                else:
                    if self.el_mot[self.pos_nuclis[i] + 3] == "-":
                        fronteres.append(self.pos_nuclis[i] + 1)
                    else:
                        fronteres.append(self.pos_nuclis[i] + 2)

            elif longi == 4:
                pos = 0

                if (pos := tros.find("s")) != -1:
                    fronteres.append(self.pos_nuclis[i] + pos + 1)
                else:
                    fronteres.append(self.pos_nuclis[i] + 2)

            elif longi == 5:
                fronteres.append(self.pos_nuclis[i] + 3)

            else:
                _LOGGER.debug(
                    f"No puc separar en sillabes el mot {self.el_mot}, cluster massa gran, de longitud {longi}"
                )
                exit(1)

        numsil = len(fronteres)
        for i in range(numsil):
            if i == 0:
                if fronteres[i] != 0:
                    esta_sil = self.el_mot[0 : fronteres[i] + 1]
                    vec_sil.append(esta_sil)
                else:
                    esta_sil = self.el_mot[0]
                    vec_sil.append(esta_sil)
            else:
                esta_sil = self.el_mot[fronteres[i - 1] + 1 : fronteres[i] + 1]
                vec_sil.append(esta_sil)

        esta_sil = self.el_mot[fronteres[numsil - 1] + 1 :]
        vec_sil.append(esta_sil)

        els_nuclis.append(self.pos_nuclis[0])
        longitud = len(vec_sil[0])

        for i in range(1, len(self.pos_nuclis)):
            this_nucli = self.pos_nuclis[i] - longitud
            els_nuclis.append(this_nucli)
            longitud += len(vec_sil[i])

        return vec_sil, els_nuclis

    def empty(self) -> bool:
        return len(self.pos_nuclis) == 0

    def mot(self) -> str:
        return self.el_mot

    def nucli(self, i: int) -> typing.Union[int, None]:
        if 0 <= i < len(self.pos_nuclis):
            return self.pos_nuclis[i]
        return None

    def size(self) -> int:
        return len(self.pos_nuclis)

    def nuclis(self) -> typing.List[int]:
        return self.pos_nuclis


class Transcripcio:
    def __init__(self, mot: str):

        self.motorig_ = mot

        self.prefixos_ = []
        self.pref_atons = []
        self.excepcions_prefs = {}
        self.excepcions_gen = set()
        self.einesgram_ = set()
        self.excep_acc = {}
        self.trossos_ = []
        self.transpart_ = []

        self.carrega_einesgram()
        self.carrega_exc_accent()

    def carrega_einesgram(self):
        # Set self.einesgram_
        self.einesgram_ = EINESGRAM

    def carrega_exc_accent(self):
        # Set self.excep_acc (excepcions d'accentuacio)
        self.excep_acc = EXCEP_ACC

    def normalize_word(self, word: str) -> str:

        word = word.lower()

        return word

    def segmenta(self, mot: str, final: typing.List[str]) -> typing.List[str]:

        # Word with prefixes segmentation

        no_te_prefix = True
        for prefix in self.prefixos_:
            lon = len(prefix)
            pos = mot.find(prefix)
            if pos != -1 and pos == 0:
                no_te_prefix = False

                if lon == len(mot):
                    final.append(mot)
                    return final
                elif lon == len(mot) - 1 and mot[lon] == "-":
                    final.append(mot)
                    return final
                else:
                    # If there are no exceptions split it
                    if prefix not in self.excepcions_prefs:
                        final.append(prefix)
                        resta = mot[lon:]
                        self.segmenta(resta, final)
                        return final
                    # If there are exceptions check that it is not part of it
                    else:
                        if mot not in self.excepcions_prefs[prefix]:
                            final.append(prefix)
                            resta = mot[lon:]
                            self.segmenta(resta, final)
                            return final
                        else:
                            final.append(mot)
                            return final

        for prefix in self.pref_atons:
            lon = len(prefix)
            pos = mot.find(prefix)
            if pos != -1 and pos == 0:
                no_te_prefix = False

                if lon == len(mot):
                    final.append(mot)
                    return final
                elif lon == len(mot) - 1 and mot[lon] == "-":
                    final.append(mot)
                    return final
                else:
                    # It should only be started if:
                    #   if the prefix ends in a vowel
                    #   only if the word continues with i, u, -r+vowel, -s+vowel
                    #   if the prefix always ends in a consonant
                    #   except in both cases
                    #   if it is part of the exceptions, if there are any
                    if acaba_en_vocal(prefix):
                        resta = mot[lon:]
                        if post_prefix_ok(resta):
                            if prefix not in self.excepcions_prefs:
                                final.append(prefix)
                                self.segmenta(resta, final)
                                return final
                            else:
                                if mot not in self.excepcions_prefs[prefix]:
                                    final.append(prefix)
                                    self.segmenta(resta, final)
                                    return final
                                else:
                                    final.append(mot)
                                    return final
                        else:
                            final.append(mot)
                            return final
                    # It is not an exception
                    else:
                        if prefix not in self.excepcions_prefs:
                            final.append(prefix)
                            queda = mot[lon:]
                            self.segmenta(queda, final)
                            return final
                        else:
                            if mot not in self.excepcions_prefs[prefix]:
                                final.append(prefix)
                                queda = mot[lon:]
                                self.segmenta(queda, final)
                                return final
                            else:
                                final.append(mot)
                                return final

        if no_te_prefix:
            final.append(mot)
            return final

    def tracta_prefixos(
        self, inici: typing.List[str], final: typing.List[str]
    ) -> typing.List[str]:

        # For each start word,
        # if there is a prefix at the beginning and the word is not part of the exception list,
        # split it after the prefix, unless after the prefix there is a hyphen

        for mot in inici:
            final = self.segmenta(mot, final)

        return final

    def parteix_mot(self):

        # Set parts
        parts = [self.motnorm_]

        self.trossos_ = self.tracta_prefixos(parts, self.trossos_)

        for tros in self.trossos_:
            partmot = Part(tros)
            self.transpart_.append(partmot)

    def no_es_nom_ment(self, mot: str) -> bool:

        if mot not in self.excepcions_gen:
            return True
        else:
            return False

    def es_adverbi(self, mot: str) -> bool:

        pos = 0
        tros = "ment"
        pos = mot.rfind(tros)
        if pos != -1:
            if pos == len(mot) - len(tros):
                if self.no_es_nom_ment(mot):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def es_exc_accent(self, mot: str) -> str:

        if mot in self.excep_acc:
            mot = self.excep_acc[mot]

        return mot

    def troba_nuclis_mot(self):

        for i in range(len(self.trossos_)):

            self.trossos_[i] = self.es_exc_accent(self.trossos_[i])

            # Determine if it's an adverb and pass the information to mot_amb_nuclis
            is_adverb = self.es_adverbi(self.trossos_[i])

            mot_amb_nuclis = MotNuclis(
                mot=self.trossos_[i],
                es_adverbi=is_adverb,
            )

            mot_amb_nuclis.troba_nuclis_mot()

            sillabes, nuclis = [], []
            if not mot_amb_nuclis.empty():
                sillabes, nuclis = mot_amb_nuclis.separa_sillabes(sillabes, nuclis)
                for sil in range(len(sillabes)):
                    sillab = Sillaba(sillabes[sil])
                    sillab.grafnuc(nuclis[sil])
                    self.transpart_[i].push_back(sillab)
            else:
                sillab = Sillaba(self.trossos_[i])
                self.transpart_[i].push_back(sillab)

    def dotze_term(self, pnum: int) -> bool:

        # retorna cert quan es mot pla (paroxiton) ja sigui per les dotze terminacions o per ser un diftong decreixent

        dift_decr = DIFT_DECR
        voc_sola = VOC_SOLA
        voc_mes_s = VOC_MES_S
        en_in = EN_IN

        numsil = self.transpart_[pnum].size()
        darsil = self.transpart_[pnum].transsil_[numsil - 1].get_text()
        darsil = darsil.lower()  # Convert to lowercase for case-insensitive comparison

        mida = len(darsil)

        # mida de la sillaba 2 o +
        if mida >= 2:
            last_dos = darsil[-2:]

            # diftong decreixent, inclou gui, qui
            for dift in dift_decr:
                es_dift_decr = last_dos == dift
                # diftong decreixent i nucli -> agut
                # diftong decreixent i no es nucli (ex: preui)-> pla
                if es_dift_decr and (
                    self.transpart_[pnum].transsil_[numsil - 1].grafnuc_ == mida - 2
                ):
                    return False
                elif es_dift_decr:
                    return True

            # vocal sola
            last_voc = darsil[-1:]
            if last_voc in voc_sola:
                return True

            # si la dar sil acaba en s (mida 2 o + encara)
            if darsil[-1:] == "s":
                if mida >= 3:
                    last_dos = darsil[-3:-1]
                    for dift in dift_decr:
                        es_dift_decr = last_dos == dift
                        if es_dift_decr and (
                            self.transpart_[pnum].transsil_[numsil - 1].grafnuc_
                            == mida - 3
                        ):
                            return False
                        elif es_dift_decr:
                            return True

                last_dos = darsil[-2:]
                if last_dos in voc_mes_s:
                    return True

            last_dos = darsil[-2:]
            if last_dos in en_in:
                return True

        last_voc = darsil[-1:]
        if last_voc in voc_sola:
            return True

        return False

    def accentua_mot(self, pnum: int):

        numsil = self.transpart_[pnum].size()

        if self.dotze_term(pnum):
            # If it ends with a vowel or vowel+s, or with o or in, it's flat (plana)
            # Vowels are aeiouàèéíòóúü
            self.transpart_[pnum].transsil_[numsil - 2].tonica()
        else:
            # Otherwise, it's acute (aguda)
            self.transpart_[pnum].transsil_[numsil - 1].tonica()

    def einagram(self, mot: str) -> bool:

        if mot not in self.einesgram_:
            return False
        else:
            return True

    def troba_accent_tonic_mot(self):

        vocaccent = ACCENTED_VOWEL_CHARS

        for pnum in range(len(self.trossos_)):

            if not self.transpart_[pnum]:
                # es una particula sense vocal
                continue

            numsil = self.transpart_[pnum].size()
            accent_grafic = False
            # bucle sobre les sillabes per veure si hi ha accent grafic
            for snum in range(numsil):
                sillaba = self.transpart_[pnum].transsil_[snum].get_text()
                if any(accented_vowel in sillaba for accented_vowel in vocaccent):

                    last_sil = self.transpart_[pnum].transsil_[numsil - 1].get_text()
                    accent_grafic = True

                    if last_sil == "ment":
                        self.transpart_[pnum].transsil_[snum].tonica()
                        self.transpart_[pnum].transsil_[numsil - 1].tonica()
                    else:
                        self.transpart_[pnum].transsil_[snum].tonica()

                    break

            if not accent_grafic:

                # si es monosillab es tonic a menys que sigui eina gramatical
                # tonic car es morfema lexematic d'una sillaba
                # si te mes d'una sillaba, estudiar la terminacio, descartant abans
                # un guio que hi pugui haver al final
                # prefixos que poden ser d'una o dues sillabes tenen nomes
                # accent secundari si son tonics i funcionen realment com a prefix

                if numsil == 1:
                    sillaba = self.transpart_[pnum].transsil_[0].get_text()
                    if self.transpart_[pnum].transsil_[0].grafnuc_ == -1:
                        # es part de mot sense nucli
                        continue
                    elif self.einagram(sillaba):
                        # amb les parts de mot
                        continue
                    else:
                        # soliem mirar si era un prefix tonic o un lexema, ja no cal
                        self.transpart_[pnum].transsil_[0].tonica()
                else:
                    # no es monosillab

                    last_sil = self.transpart_[pnum].transsil_[numsil - 1].get_text()
                    # no es referencia, last_sil, car volem guardar el valor

                    if last_sil == "ment":
                        # no cal tractar diferent els prefixos tonics
                        if self.no_es_nom_ment(
                            self.trossos_[pnum]
                        ) and self.no_es_nom_ment(self.motnorm_):
                            if numsil - 1 > 1:
                                self.transpart_[
                                    pnum
                                ].pop_back()  # Remove the last syllable
                                self.accentua_mot(pnum)  # Accentuate from the syllables
                                darsil = Sillaba(
                                    last_sil
                                )  # Create a syllable like before
                                self.transpart_[pnum].push_back(
                                    darsil
                                )  # Add it and make it tonic
                                self.transpart_[pnum].transsil_[numsil - 1].tonica()
                                self.transpart_[pnum].transsil_[numsil - 1].grafnuc_ = 1
                                # # es la e de ment
                            else:
                                self.transpart_[pnum].transsil_[0].tonica()
                                self.transpart_[pnum].transsil_[numsil - 1].tonica()
                    else:
                        self.accentua_mot(pnum)

    def sillaba_accentua_mot(self):

        self.parteix_mot()
        self.troba_nuclis_mot()
        self.troba_accent_tonic_mot()

    def stress_tonic(self) -> str:

        accent_changes = ACCENT_CHANGES

        all_vowels = VOWEL_CHARS
        accented_vowels = ACCENTED_VOWEL_CHARS
        unaccented_vowels = list(set(all_vowels) - set(accented_vowels))

        original_word = ""
        stressed_word = ""

        for i in range(len(self.transpart_)):

            word = self.transpart_[i].text_

            if any(ext in word for ext in accented_vowels):
                stressed_word = stressed_word + word
            else:
                for j in range(self.transpart_[i].size()):
                    sil = self.transpart_[i].transsil_[j]
                    sillaba_text = sil.get_text()
                    idxgrafnucli = sil.get_grafnuc()
                    is_tonic = sil.es_sil_tonica()

                    if is_tonic == "sí":
                        sillaba_list = list(sillaba_text)
                        if sillaba_list[idxgrafnucli] in unaccented_vowels:
                            if sillaba_list[idxgrafnucli] == "e":
                                if j == self.transpart_[i].size() - 1:
                                    # for accute words almost always this is the correct accented e
                                    sillaba_list[idxgrafnucli] = "è"
                                elif j == self.transpart_[i].size() - 2:
                                    # the word has its accent in the penultimate sillabe
                                    # almost always this is the correct accented e
                                    sillaba_list[idxgrafnucli] = "è"
                                else:
                                    # proparoxytone
                                    # almost always this is the correct accented e
                                    sillaba_list[idxgrafnucli] = "è"
                            elif sillaba_list[idxgrafnucli] == "o":
                                if j == self.transpart_[i].size() - 1:
                                    # the word has its accent in the last sillabe
                                    # almost always this is the correct accented o
                                    sillaba_list[idxgrafnucli] = "ó"
                                elif j == self.transpart_[i].size() - 2:
                                    # the word has its accent in the penultimate sillabe
                                    # almost always this is the correct accented o
                                    sillaba_list[idxgrafnucli] = "ò"
                                else:
                                    # proparoxytone
                                    # almost always this is the correct accented o
                                    sillaba_list[idxgrafnucli] = "ò"
                            else:
                                sillaba_list[idxgrafnucli] = accent_changes[
                                    sillaba_list[idxgrafnucli]
                                ]

                            sillaba_text = "".join(sillaba_list)

                    stressed_word = stressed_word + sillaba_text

            original_word = original_word + word

        return stressed_word

    def stress_word(self) -> str:

        self.motnorm_ = self.normalize_word(self.motorig_)

        self.sillaba_accentua_mot()

        self.stressed_word = self.stress_tonic()

        return self.stressed_word


class CatalanPreProcessText:
    """Pre-processes text"""

    # The preprocessing is the same for all accents in this version (variable lang is not used)

    def __init__(self, lookup_phonemes, settings_values: dict, lang: str):

        self.lookup_phonemes = lookup_phonemes
        self.settings_values = settings_values
        self.lang = lang

    def __call__(self, text: str) -> str:

        breaks = [" "]
        breaks = breaks + list(self.settings_values["major_breaks"])
        breaks = breaks + list(self.settings_values["minor_breaks"])
        breaks = breaks + list(self.settings_values["word_breaks"])
        breaks = breaks + list(self.settings_values["begin_punctuations"])
        breaks = breaks + list(self.settings_values["end_punctuations"])

        tokens = [text.strip()]
        for char_break in breaks:
            sub_tokens = [re.split(f"(\\{char_break})", item) for item in tokens]
            tokens = [item for sublist in sub_tokens for item in sublist if item != ""]

        preprocessed_tokens = []
        for token in tokens:

            try:
                if token in breaks:
                    processed_token = token
                else:
                    is_in_lexicon = self.lookup_phonemes(token) is not None
                    if is_in_lexicon:
                        processed_token = token
                    else:
                        tr = Transcripcio(token)
                        processed_token = tr.stress_word()
            except Exception:
                processed_token = token
                _LOGGER.debug("Unable to stress token %s.", token)

            preprocessed_tokens.append(processed_token)

        processed_text = "".join(preprocessed_tokens)

        _LOGGER.debug("%s preprocessed obtaining: %s", text, processed_text)

        return processed_text


# Post-Process constants
# Only defined for "ca", "ca-ce" accent.
# For the rest of accents, not post-processing is done

PHONEME_VOWELS = ["'a", "'ɛ", "'ɔ", "'e", "'i", "'o", "'u", "ə", "i", "u"]
PHONEME_STRESSED_VOWELS = ["'a", "'ɛ", "'ɔ", "'e", "'i", "'o", "'u"]
PHONEME_HIGH_VOWELS = ["i", "u", "'i", "'u"]
PHONEME_NEUTRAL_VOWELS = ["ə"]

# Post-Process functions and classes


def identify_lang(
    nodes: typing.List[
        typing.Union[WordNode, BreakWordNode, BreakNode, PunctuationWordNode]
    ]
) -> str:

    try:
        for node in nodes:
            if isinstance(node, WordNode):
                lang = node.lang
                break
    except Exception:
        lang = "ca"

    return lang


def phoneme_is_vowel(phoneme: str) -> bool:
    return phoneme in PHONEME_VOWELS


def phoneme_is_stressed_vowel(phoneme: str) -> bool:
    return phoneme in PHONEME_STRESSED_VOWELS


def phoneme_is_unstressed_vowel(phoneme: str) -> bool:
    return phoneme_is_vowel(phoneme) and not phoneme_is_stressed_vowel(phoneme)


def phoneme_is_high_vowel(phoneme: str) -> bool:
    return phoneme in PHONEME_HIGH_VOWELS


def phoneme_is_high_stressed_vowel(phoneme: str) -> bool:
    return phoneme_is_high_vowel(phoneme) and phoneme_is_stressed_vowel(phoneme)


def phoneme_is_high_unstressed_vowel(phoneme: str) -> bool:
    return phoneme_is_high_vowel(phoneme) and phoneme_is_unstressed_vowel(phoneme)


def phoneme_is_neutral_vowel(phoneme: str) -> bool:
    return phoneme in PHONEME_NEUTRAL_VOWELS


def fusion_if_needed(node_1: WordNode, node_2: WordNode, lang: str):

    if lang in ["ca", "ca-ce"]:
        if len(node_1.phonemes) == 0 or len(node_2.phonemes) == 0:
            return
        else:

            last_phoneme_word_1 = node_1.phonemes[-1]
            first_phoneme_word_2 = node_2.phonemes[0]

            # Case 1: high unstressed vowel + stressed vowel of the same timbre
            if (
                phoneme_is_high_unstressed_vowel(last_phoneme_word_1)
                and phoneme_is_high_stressed_vowel(first_phoneme_word_2)
                and last_phoneme_word_1 == first_phoneme_word_2.replace("'", "")
            ):
                # Case [i] + [i'] = [i'] or [u] + [u'] = [u']
                node_1.phonemes.pop()
                _LOGGER.debug(
                    f"FUSION CASE 1 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                )

            # Case 2: high unstressed vowel + high unstressed vowel of the same timbre
            elif (
                phoneme_is_high_unstressed_vowel(last_phoneme_word_1)
                and phoneme_is_high_unstressed_vowel(first_phoneme_word_2)
                and last_phoneme_word_1 == first_phoneme_word_2
            ):
                # Case [i] + [i] = [i] or [u] + [u] = [u]
                node_1.phonemes.pop()
                _LOGGER.debug(
                    f"FUSION CASE 2 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                )

            # Case 3: neutral vowel + neutral vowel (except if any of the vowels is the proposition "a")
            elif (
                phoneme_is_neutral_vowel(last_phoneme_word_1)
                and phoneme_is_neutral_vowel(first_phoneme_word_2)
                and node_1.text != "a"
                and node_2.text != "a"
            ):
                node_1.phonemes.pop()
                _LOGGER.debug(
                    f"FUSION CASE 3 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                )
    else:
        pass


def elision_if_needed(node_1: WordNode, node_2: WordNode, lang: str):

    if lang in ["ca", "ca-ce"]:

        if len(node_1.phonemes) == 0 or len(node_2.phonemes) == 0:
            return
        else:

            last_phoneme_word_1 = node_1.phonemes[-1]
            first_phoneme_word_2 = node_2.phonemes[0]

            # Case 1: stressed vowel ['a], ['ɛ], ['e], ['o] or ['ɔ] + neutral vowel (except if any of the vowels is the proposition "a")
            if (
                phoneme_is_stressed_vowel(last_phoneme_word_1)
                and not phoneme_is_high_vowel(last_phoneme_word_1)
            ) and (
                phoneme_is_neutral_vowel(first_phoneme_word_2) and node_2.text != "a"
            ):
                node_2.phonemes.pop(0)
                _LOGGER.debug(
                    f"ELISION CASE 1 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                )

            # Case 2: neutral vowel + stressed vowel ['a], ['ɛ], ['e], ['o] or ['ɔ]
            elif phoneme_is_neutral_vowel(last_phoneme_word_1) and (
                phoneme_is_stressed_vowel(first_phoneme_word_2)
                and not phoneme_is_high_vowel(first_phoneme_word_2)
            ):
                node_1.phonemes.pop()
                _LOGGER.debug(
                    f"ELISION CASE 2 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                )
    else:
        pass


def diphthong_if_needed(node_1: WordNode, node_2: WordNode, lang: str):

    if lang in ["ca", "ca-ce"]:

        if len(node_1.phonemes) == 0 or len(node_2.phonemes) == 0:
            return
        else:

            last_phoneme_word_1 = node_1.phonemes[-1]
            first_phoneme_word_2 = node_2.phonemes[0]

            # Case 1: stressed vowel + high unstressed vowel
            if (
                phoneme_is_stressed_vowel(last_phoneme_word_1)
                and not phoneme_is_high_vowel(last_phoneme_word_1)
            ) and phoneme_is_high_unstressed_vowel(first_phoneme_word_2):
                if first_phoneme_word_2 == "i":
                    # Case [stressed vowel] + [i] = [stressed vowel + j], stressed vowel not 'i or 'u
                    node_2.phonemes[0] = "j"
                    _LOGGER.debug(
                        f"DIPTHONG CASE 1 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                    )

                elif first_phoneme_word_2 == "u":
                    # Case [stressed vowel] + [u] = [stressed vowel + uw], stressed vowel not 'i or 'u
                    node_2.phonemes[0] = "uw"
                    _LOGGER.debug(
                        f"DIPTHONG CASE 1 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                    )

            # Case 2: high unstressed vowel + stressed vowel
            elif phoneme_is_high_unstressed_vowel(
                last_phoneme_word_1
            ) and phoneme_is_stressed_vowel(first_phoneme_word_2):
                if (
                    last_phoneme_word_1 == "i"
                    and first_phoneme_word_2 not in ["'i"]
                    and node_1.text in ["hi", "ho", "i"]
                ):
                    # Case [i] + [stressed] = [y + stressed vowel], i only from "hi", "ho" or "i"
                    node_1.phonemes[-1] = "y"
                    _LOGGER.debug(
                        f"DIPTHONG CASE 2 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                    )

                elif (
                    last_phoneme_word_1 == "u"
                    and first_phoneme_word_2 not in ["'u"]
                    and node_1.text in ["hi", "ho", "i"]
                ):
                    # Case [u] + [stressed] = [u + stressed vowel], i only from "hi", "ho" or "i"
                    pass

            # Case 3: unstressed vowel + high unstressed vowel
            elif phoneme_is_neutral_vowel(
                last_phoneme_word_1
            ) and phoneme_is_high_unstressed_vowel(first_phoneme_word_2):
                if first_phoneme_word_2 == "i":
                    # Case [neutral vowel] + [i] = [neutral vowel + j]
                    node_2.phonemes[0] = "j"
                    _LOGGER.debug(
                        f"DIPTHONG CASE 3 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                    )

                elif first_phoneme_word_2 == "u":
                    # Case [neutral vowel] + [u] = [neutral vowel + uw]
                    node_2.phonemes[0] = "uw"
                    _LOGGER.debug(
                        f"DIPTHONG CASE 3 {node_1.text} {node_2.text}: {node_1.phonemes} {node_2.phonemes}"
                    )

            # Case 4: unstressed vowel + high unstressed vowel
            elif phoneme_is_high_unstressed_vowel(
                last_phoneme_word_1
            ) and phoneme_is_neutral_vowel(first_phoneme_word_2):
                pass
    else:
        pass


def ca_post_process_sentence(
    graph: GraphType, sent_node: SentenceNode, settings: TextProcessorSettings
):

    # Create a list of relevant nodes
    nodes = []
    for dfs_node in nx.dfs_preorder_nodes(graph, sent_node.node):

        node = graph.nodes[dfs_node][DATA_PROP]

        if not graph.out_degree(dfs_node) == 0:
            # Only leave
            continue

        node = graph.nodes[dfs_node][DATA_PROP]
        if isinstance(node, WordNode):
            nodes.append(typing.cast(WordNode, node))
        if isinstance(node, BreakWordNode):
            nodes.append(typing.cast(BreakWordNode, node))
        if isinstance(node, BreakNode):
            nodes.append(typing.cast(BreakNode, node))
        if isinstance(node, PunctuationWordNode):
            nodes.append(typing.cast(PunctuationWordNode, node))

    lang = identify_lang(nodes)

    # HACK
    # Training corpora includes an invalid sequence of phonemes: l ʎ l
    # We fix that here, in the next iteration will be properly solved
    phonemes_to_fix = "l ʎ l"
    fixed_phonemes = "l l"
    for node in nodes:

        if node is None:
            continue

        if isinstance(node, WordNode):
            if not (node.text and node.phonemes):
                continue
            phonemes_text = " ".join(node.phonemes)
            if phonemes_to_fix in phonemes_text:
                phonemes_text = phonemes_text.replace(phonemes_to_fix, fixed_phonemes)
                node.phonemes = phonemes_text.split(" ")
                _LOGGER.debug(
                    f"FIX: phoneme sequence '{phonemes_to_fix}' fixed at {node.text}. Fixed transcription: {node.phonemes}"
                )

    # Create a list of contiguous word nodes
    contiguous_word_nodes = []
    for node_1, node_2 in sliding_window(nodes, 2):

        if node_1 is None or node_2 is None:
            continue

        if isinstance(node_1, WordNode) and isinstance(node_2, WordNode):
            if not (
                node_1.text and node_1.phonemes and node_2.text and node_2.phonemes
            ):
                continue
            contiguous_word_nodes.append([node_1, node_2])

    for (node_1, node_2) in contiguous_word_nodes:

        diphthong_if_needed(node_1, node_2, lang)
        fusion_if_needed(node_1, node_2, lang)
        elision_if_needed(node_1, node_2, lang)


# Settings


def get_ca_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:

    """Create settings for Catalan"""

    try:
        lang = str(lang_dir).split("/")[-1]
        main_lang, lang_version = lang.split("-")
        lang = f"{main_lang.lower()}-{lang_version.upper()}"
    except Exception:
        lang = "ca"

    lookup_phonemes = settings_args["lookup_phonemes"]

    settings_values = {
        "major_breaks": {".", "?", "!"},
        "minor_breaks": {",", ";", ":", "..."},
        "word_breaks": {"_"},
        "begin_punctuations": {'"', "“", "«", "[", "(", "<", "¡", "¿"},
        "end_punctuations": {'"', "”", "»", "]", ")", ">", "!", "?"},
        "default_currency": "EUR",
        "default_date_format": InterpretAsFormat.DATE_DMY,
        "replacements": [
            ("’", "'"),  # normalize apostrophe
            ("'", ""),  # remove orthographic apostrophe
            ("-", ""),
            ("l·l", "l"),
        ],
    }

    settings_args = {
        **settings_values,
        "pre_process_text": CatalanPreProcessText(
            lookup_phonemes, settings_values, lang
        ),
        "post_process_sentence": ca_post_process_sentence,
        **settings_args,
    }

    return TextProcessorSettings(lang="ca", **settings_args)


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
        self, word: str, role: typing.Optional[str] = None, do_transforms: bool = True
    ) -> typing.Optional[PHONEMES_TYPE]:
        if self.phonemizer is None:
            _LOGGER.debug("Connecting to lexicon database at %s", self.db_path)
            db_conn = sqlite3.connect(str(self.db_path))
            self.phonemizer = SqlitePhonemizer(db_conn=db_conn, **self.phonemizer_args)

        assert self.phonemizer is not None
        return self.phonemizer(word, role=role, do_transforms=do_transforms)
