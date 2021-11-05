#!/usr/bin/env python3
"""Language-specific settings"""
import logging
import re
import sqlite3
import typing
from pathlib import Path

import networkx as nx

from gruut.const import PHONEMES_TYPE, GraphType, SentenceNode, Time
from gruut.g2p import GraphemesToPhonemes
from gruut.phonemize import SqlitePhonemizer
from gruut.pos import PartOfSpeechTagger
from gruut.text_processor import InterpretAsFormat, TextProcessorSettings
from gruut.utils import find_lang_dir, remove_non_word_chars, resolve_lang

_LOGGER = logging.getLogger("gruut")

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
                    "(%s) no part of speech tagger found at %s", lang, pos_model_path,
                )

        # Phonemizer
        if load_phoneme_lexicon and ("lookup_phonemes" not in settings_args):
            lexicon_db_path = lang_dir / lang_model_prefix / "lexicon.db"
            if lexicon_db_path.is_file():
                # Transformations to apply to words when they can't be found in the lexicon
                phonemizer_args = {
                    "word_transform_funcs": [
                        remove_non_word_chars,
                        lambda s: remove_non_word_chars(s.lower()),
                    ],
                    "casing_func": str.lower,
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
         \s*(a\\.m\\.|am|pm|p\\.m\\.|a\\.m|p\\.m)? # am/pm
         $""",
    re.IGNORECASE | re.X,
)


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
    past_noon = hour >= 12

    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12
        past_noon = True

    yield str(hour)

    minute = time.minutes
    if minute > 0:
        if minute < 10:
            yield "oh"

        yield str(minute)

    if time.period is None:
        if past_noon:
            yield "P.M."
        else:
            yield "A.M."
    else:
        yield time.period


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
        "replacements": [("’", "'")],  # normalize apostrophe
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
