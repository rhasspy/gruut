"""gruut module"""
import itertools
import sqlite3
import re
import threading
import typing
import logging
from enum import Enum
from pathlib import Path

import gruut_ipa

import networkx as nx
from gruut.lang import KNOWN_LANGS, resolve_lang
from gruut.text_processor import (
    TextProcessor,
    TextProcessorSettings,
    GetPartsOfSpeech,
    Sentence,
)
from gruut.pos import PartOfSpeechTagger
from gruut.g2p import GraphemesToPhonemes
from gruut.phonemize import SqlitePhonemizer
from gruut.utils import find_lang_dir
from gruut.const import PHONEMES_TYPE

# from .const import WORD_PHONEMES, Sentence, Token, TokenFeatures, WordPronunciation
# from .lang import KNOWN_LANGS, get_phonemizer, get_tokenizer, resolve_lang
# from .phonemize import Phonemizer, SqlitePhonemizer, UnknownWordError
# from .toksen import RegexTokenizer, Tokenizer
# from .utils import encode_inline_pronunciations

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("gruut")

__version__ = (_DIR / "VERSION").read_text().strip()
__author__ = "Michael Hansen (synesthesiam)"
__all__ = [
    "sentences",
    "get_text_processor",
    "is_language_supported",
    "get_supported_languages",
]

# -----------------------------------------------------------------------------

SETTINGS: typing.Dict[str, TextProcessorSettings] = {}
SETTINGS_LOCK = threading.RLock()


def sentences(
    text: str,
    lang: str = "en_US",
    ssml: bool = False,
    espeak: bool = False,
    major_breaks: bool = True,
    minor_breaks: bool = True,
    punctuations: bool = True,
    **process_args,
) -> typing.Iterable[Sentence]:
    model_prefix = "" if (not espeak) else "espeak"
    settings = {}

    with SETTINGS_LOCK:
        langs_to_load = {lang}

        if ssml:
            # Need to load all languages for SSML since the 'lang' attribute can
            # be used almost anywhere.
            langs_to_load.update(KNOWN_LANGS)

        for load_lang in langs_to_load:
            load_lang = resolve_lang(load_lang)

            if model_prefix:
                lang_with_prefix = f"{load_lang}/{model_prefix}"
            else:
                lang_with_prefix = load_lang

            if lang_with_prefix not in SETTINGS:
                SETTINGS[lang_with_prefix] = get_settings(
                    load_lang, model_prefix=model_prefix
                )

            settings[load_lang] = SETTINGS[lang_with_prefix]

            if "-" in load_lang:
                # Add en_US as an alias for en-us
                lang_parts = load_lang.split("-", maxsplit=1)
                underscore_lang = f"{lang_parts[0]}_{lang_parts[1].upper()}"
                settings[underscore_lang] = SETTINGS[load_lang]

    text_processor = TextProcessor(default_lang=lang, settings=settings)
    graph, root = text_processor(text, ssml=ssml, **process_args)

    yield from text_processor.sentences(
        graph,
        root,
        major_breaks=major_breaks,
        minor_breaks=minor_breaks,
        punctuations=punctuations,
    )


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
    lang = resolve_lang(lang)

    if lang_dir is None:
        # Search for language data files
        lang_dir = find_lang_dir(lang, search_dirs=search_dirs)

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
                # TODO: Remove non-word chars in locale
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

    if lang == "ar":
        # Arabic
        return make_ar_settings(lang_dir, **settings_args)

    if lang in {"en-us", "en-gb"}:
        # English
        return make_en_us_settings(lang_dir, **settings_args)

    if lang == "de-de":
        # German
        return make_de_settings(lang_dir, **settings_args)

    if lang == "fr-fr":
        # French
        return make_fr_settings(lang_dir, **settings_args)

    if lang == "es-es":
        # Spanish
        return make_es_settings(lang_dir, **settings_args)

    if lang == "nl":
        # Dutch
        return make_nl_settings(lang_dir, **settings_args)

    if lang == "it-it":
        # Italian
        return make_it_settings(lang_dir, **settings_args)

    # Default settings only
    return TextProcessorSettings(lang=lang, **settings_args)


# -----------------------------------------------------------------------------


def get_text_processor(
    default_lang: str = "en_US",
    languages: typing.Optional[typing.Iterable[str]] = None,
    lang_dir: typing.Optional[typing.Union[str, Path]] = None,
    search_dirs: typing.Optional[typing.Iterable[typing.Union[str, Path]]] = None,
    model_prefix: str = "",
    load_pos_tagger: bool = True,
    load_phoneme_lexicon: bool = True,
    load_g2p_guesser: bool = True,
    **settings_args,
) -> TextProcessor:
    if languages is None:
        languages = itertools.chain(KNOWN_LANGS, [default_lang])

    if not languages:
        raise ValueError("languages must not be empty")

    settings = {}

    for language in languages:
        if language in settings:
            continue

        # Resolve language
        if model_prefix:
            # espeak
            lang_model_prefix = model_prefix
            language_only = language
        elif "/" in language:
            # en-us/espeak
            language_only, lang_model_prefix = language.split("/", maxsplit=1)
        else:
            # en-us
            language_only = language
            lang_model_prefix = ""

        # en_US -> en-us
        language_only = resolve_lang(language_only)

        if lang_model_prefix:
            lang_with_prefix = f"{language_only}/{lang_model_prefix}"
        else:
            lang_with_prefix = language_only

        if lang_with_prefix not in settings:
            # Create settings
            settings[lang_with_prefix] = get_settings(
                lang=language_only,
                model_prefix=lang_model_prefix,
                search_dirs=search_dirs,
                lang_dir=lang_dir,
                load_pos_tagger=load_pos_tagger,
                load_phoneme_lexicon=load_phoneme_lexicon,
                load_g2p_guesser=load_g2p_guesser,
            )

        # Mirror settings for original language form (e.g., en_US)
        settings[language] = settings[lang_with_prefix]

    return TextProcessor(default_lang=default_lang, settings=settings,)


# -----------------------------------------------------------------------------
# Arabic
# -----------------------------------------------------------------------------

DEFAULT_AR_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "؟", "!"},
    "minor_breaks": {"،", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">"},
    "default_date_format": "omy",  # 4/1/2021 -> first April twenty twenty-one
    "replacements": [
        ("’", "'"),  # normalize apostrophe
        ("\\B['‘]", '"'),  # replace single quotes (left)
        ("['’]\\B", '"'),  # replace signed quotes (right)
    ],
}


def make_ar_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_AR_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="ar", **settings_args)


# -----------------------------------------------------------------------------
# English
# -----------------------------------------------------------------------------


# TTS and T.T.S.
EN_INITIALISM_PATTERN = re.compile(r"^[A-Z]{2,}$")
EN_INITIALISM_DOTS_PATTERN = re.compile(r"^(?:[a-zA-Z]\.){2,}$")


def en_is_initialism(text: str) -> bool:
    """True if text is of the form TTS or T.T.S."""
    return (EN_INITIALISM_PATTERN.match(text) is not None) or (
        EN_INITIALISM_DOTS_PATTERN.match(text) is not None
    )


DEFAULT_EN_US_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "?", "!"},
    "minor_breaks": {",", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">"},
    "default_currency": "USD",
    "default_date_format": "moy",  # 4/1/2021 -> April first twenty twenty-one
    "is_initialism": en_is_initialism,
    "split_initialism": lambda text: list(text.replace(".", "")),
    "replacements": [
        ("\\B'", '"'),  # replace single quotes (left)
        ("'\\B", '"'),  # replace signed quotes (right)
    ],
    "abbreviations": {
        r"^([cC])o\.": "\1ompany",  # co. -> company
        r"^([dD])r\.": "\1octor",  # dr. -> doctor
        r"^([dD])rs\.": "\1octors",  # drs. -> doctors
        r"^([jJ])r\.": "\1unior",  # jr. -> junior
        r"^([lL])td\.": "\1imited",  # -> ltd. -> limited
        r"^([mM])r\.": "\1ister",  # -> mr. -> mister
        r"^([mM])s\.": "\1iss",  # -> ms. -> miss
        r"^([mM])rs\.": "\1isess",  # -> mrs. -> misess
        r"^([sS])t\.": "\1treet",  # -> st. -> street
        r"(.*\d)%": "\1 percent",  # % -> percent
        r"^&$": "and",  # &-> and
    },
    "spell_out_words": {
        ".": "dot",
        "-": "dash",
        "@": "at",
        "*": "star",
        "+": "plus",
        "/": "slash",
    },
}


def make_en_us_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_EN_US_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="en_US", **settings_args)


# -----------------------------------------------------------------------------
# German
# -----------------------------------------------------------------------------

DEFAULT_DE_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "?", "!"},
    "minor_breaks": {",", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<", "’", "„"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">", "’"},
    "default_currency": "EUR",
    "default_date_format": "omy",  # 4/1/2021 -> first April twenty twenty-one
    "replacements": [
        ("’", "'"),  # normalize apostrophe
        ("\\B['‘]", '"'),  # replace single quotes (left)
        ("['’]\\B", '"'),  # replace signed quotes (right)
    ],
}


def make_de_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_DE_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="de_DE", **settings_args)


# -----------------------------------------------------------------------------
# French
# -----------------------------------------------------------------------------


def fr_post_process_sentence(graph, sent_node, settings):
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


DEFAULT_FR_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "?", "!"},
    "minor_breaks": {",", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">"},
    "default_currency": "EUR",
    "default_date_format": "omy",  # 4/1/2021 -> first April twenty twenty-one
    "replacements": [
        ("’", "'"),  # normalize apostrophe
        ("\\B['‘]", '"'),  # replace single quotes (left)
        ("['’]\\B", '"'),  # replace signed quotes (right)
    ],
    "post_process_sentence": fr_post_process_sentence,
}


def make_fr_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_FR_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="fr_FR", **settings_args)


# -----------------------------------------------------------------------------
# Spanish
# -----------------------------------------------------------------------------

DEFAULT_ES_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "?", "!"},
    "minor_breaks": {",", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<", "¡", "¿"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">"},
    "default_currency": "EUR",
    "default_date_format": "omy",  # 4/1/2021 -> first April twenty twenty-one
    "replacements": [
        ("’", "'"),  # normalize apostrophe
        ("\\B['‘]", '"'),  # replace single quotes (left)
        ("['’]\\B", '"'),  # replace signed quotes (right)
    ],
}


def make_es_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_ES_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="es_ES", **settings_args)


# -----------------------------------------------------------------------------
# Dutch
# -----------------------------------------------------------------------------

DEFAULT_NL_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "?", "!"},
    "minor_breaks": {",", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">"},
    "default_currency": "EUR",
    "default_date_format": "omy",  # 4/1/2021 -> first April twenty twenty-one
    "replacements": [
        ("’", "'"),  # normalize apostrophe
        ("\\B['‘]", '"'),  # replace single quotes (left)
        ("['’]\\B", '"'),  # replace signed quotes (right)
    ],
}


def make_nl_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_NL_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="nl", **settings_args)


# -----------------------------------------------------------------------------
# Italian
# -----------------------------------------------------------------------------

DEFAULT_IT_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "?", "!"},
    "minor_breaks": {",", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<", "„"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">"},
    "default_currency": "EUR",
    "default_date_format": "omy",  # 4/1/2021 -> first April twenty twenty-one
    "replacements": [
        ("’", "'"),  # normalize apostrophe
        ("\\B['‘]", '"'),  # replace single quotes (left)
        ("['’]\\B", '"'),  # replace signed quotes (right)
    ],
    "post_process_sentence": fr_post_process_sentence,
}


def make_it_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_IT_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="it_IT", **settings_args)


# -----------------------------------------------------------------------------


def is_language_supported(lang: str) -> bool:
    """True if gruut supports lang"""
    return resolve_lang(lang) in KNOWN_LANGS


def get_supported_languages() -> typing.Set[str]:
    """Set of supported gruut languages"""
    return set(KNOWN_LANGS)


# -----------------------------------------------------------------------------


class DelayedPartOfSpeechTagger:
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


class DelayedGraphemesToPhonemes:
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
