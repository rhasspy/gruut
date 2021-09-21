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

# -----------------------------------------------------------------------------

_SETTINGS: typing.Dict[str, TextProcessorSettings] = {}
_SETTINGS_LOCK = threading.RLock()


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

    with _SETTINGS_LOCK:
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

            if lang_with_prefix not in _SETTINGS:
                _SETTINGS[lang_with_prefix] = get_settings(
                    load_lang, model_prefix=model_prefix
                )

            settings[lang] = _SETTINGS[lang_with_prefix]

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

    if lang in {"en-us", "en-gb"}:
        # English
        return make_en_US_settings(lang_dir, **settings_args)

    # Default settings only
    return TextProcessorSettings(lang=lang)


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
            language = f"{language_only}/{lang_model_prefix}"
        else:
            language = language_only

        if language not in settings:
            settings[language] = get_settings(
                lang=language_only,
                model_prefix=lang_model_prefix,
                search_dirs=search_dirs,
                lang_dir=lang_dir,
                load_pos_tagger=load_pos_tagger,
                load_phoneme_lexicon=load_phoneme_lexicon,
                load_g2p_guesser=load_g2p_guesser,
            )

    return TextProcessor(default_lang=default_lang, settings=settings,)


# -----------------------------------------------------------------------------


# TTS and T.T.S.
EN_INITIALISM_PATTERN = re.compile(r"^[A-Z]{2,}$")
EN_INITIALISM_DOTS_PATTERN = re.compile(r"^(?:[a-zA-Z]\.){2,}$")

DEFAULT_EN_US_SETTINGS: typing.Dict[str, typing.Any] = {
    "major_breaks": {".", "?", "!"},
    "minor_breaks": {",", ";", ":"},
    "word_breaks": {"-", "_"},
    "begin_punctuations": {'"', "“", "«", "[", "(", "<"},
    "end_punctuations": {'"', "”", "»", "]", ")", ">"},
    "default_currency": "USD",
    "default_date_format": "moy",  # 4/1/2021 -> April first twenty twenty-one
    "is_initialism": lambda text: (EN_INITIALISM_PATTERN.match(text) is not None)
    or (EN_INITIALISM_DOTS_PATTERN.match(text) is not None),
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


def make_en_US_settings(lang_dir=None, **settings_args) -> TextProcessorSettings:
    settings_args = {**DEFAULT_EN_US_SETTINGS, **settings_args}
    return TextProcessorSettings(lang="en_US", **settings_args)


# class TextToPhonemesReturn(str, Enum):
#     """
#     Format of return value from :py:meth:`~gruut.text_to_phonemes`.
#     """

#     WORD_TUPLES = "word_tuples"
#     """Tuples of the form (sentence index, word, word phonemes)"""

#     FLAT_PHONEMES = "flat_phonemes"
#     """Flat list of phoneme strings"""

#     WORD_PHONEMES = "word_phonemes"
#     """Lists of phonemes grouped by words only"""

#     SENTENCE_PHONEMES = "sentence_phonemes"
#     """Lists of phonemes grouped by sentence only"""

#     SENTENCE_WORD_PHONEMES = "sentence_word_phonemes"
#     """Lists of phonemes grouped by sentence, then word"""

#     SENTENCES = "sentences"
#     """List of :py:class:`~gruut.const.Sentence` objects with tokens, features, etc."""


# # (sentence index, word, [phonemes])
# WORD_TUPLES_TYPE = typing.List[typing.Tuple[int, str, WORD_PHONEMES]]
# FLAT_PHONEMES_TYPE = typing.List[str]
# WORD_PHONEMES_TYPE = typing.List[WORD_PHONEMES]
# SENTENCE_PHONEMES_TYPE = typing.List[WORD_PHONEMES]
# SENTENCE_WORD_PHONEMES_TYPE = typing.List[typing.List[WORD_PHONEMES]]
# SENTENCES_TYPE = typing.List[Sentence]


# TEXT_TO_PHONEMES_RETURN_TYPE = typing.Union[
#     WORD_TUPLES_TYPE,
#     FLAT_PHONEMES_TYPE,
#     WORD_PHONEMES_TYPE,
#     SENTENCE_PHONEMES_TYPE,
#     SENTENCE_WORD_PHONEMES_TYPE,
#     SENTENCES_TYPE,
# ]

# -----------------------------------------------------------------------------


# def text_to_phonemes(
#     text: str,
#     lang: str = "en-us",
#     return_format: typing.Union[str, TextToPhonemesReturn] = "word_tuples",
#     no_cache: bool = False,
#     inline_pronunciations: typing.Optional[bool] = None,
#     tokenizer: typing.Optional[Tokenizer] = None,
#     tokenizer_args: typing.Optional[typing.Mapping[str, typing.Any]] = None,
#     phonemizer: typing.Optional[Phonemizer] = None,
#     phonemizer_args: typing.Optional[typing.Mapping[str, typing.Any]] = None,
# ) -> TEXT_TO_PHONEMES_RETURN_TYPE:
#     """
#     High-level text to phonemes interface.

#     Args:
#         text: Text to tokenize and phonemize
#         lang: Language of the text
#         return_format: Format of return value
#         no_cache: If True, tokenizer/phonemizer cache are not used
#         inline_pronunciations: If True, allow inline [[ p h o n e m e s ]] and {{ w{or}ds }} in text
#         tokenizer: Optional tokenizer to use instead of creating one
#         tokenizer_args: Optional keyword arguments used when creating tokenizer
#         phonemizer: Optional phonemizer to use instead of creating one
#         phonemizer_args: Optional keyword arguments used when creating phonemizer

#     Returns:
#         Tuples of the form (sentence index, word, [word phonemes]). See TextToPhonemesReturn enum for more return formats.

#     Example:
#         ::

#             from gruut import text_to_phonemes

#             text = 'He wound it around the wound, saying "I read it was $10 to read."'

#             for sent_idx, word, word_phonemes in text_to_phonemes(text, lang="en-us"):
#                 print(word, *word_phonemes)

#         Output::

#             he h ˈi
#             wound w ˈaʊ n d
#             it ˈɪ t
#             around ɚ ˈaʊ n d
#             the ð ə
#             wound w ˈu n d
#             , |
#             saying s ˈeɪ ɪ ŋ
#             i ˈaɪ
#             read ɹ ˈɛ d
#             it ˈɪ t
#             was w ə z
#             ten t ˈɛ n
#             dollars d ˈɑ l ɚ z
#             to t ə
#             read ɹ ˈi d
#             . ‖
#     """
#     global _TOKENIZER_CACHE_ARGS, _PHONEMIZER_CACHE_ARGS

#     lang = resolve_lang(lang)

#     if inline_pronunciations:
#         with _CACHE_LOCK:
#             # Load phonemes for language
#             lang_phonemes = _PHONEMES_CACHE.get(lang)

#             if lang_phonemes is None:
#                 lang_phonemes = gruut_ipa.Phonemes.from_language(lang)
#                 assert lang_phonemes is not None, f"Unsupported language: {lang}"
#                 _PHONEMES_CACHE[lang] = lang_phonemes

#         assert lang_phonemes is not None

#         # Replace [[ p h o n e m e s ]] with __phonemes_<base32-phonemes>__
#         text = encode_inline_pronunciations(text, lang_phonemes)

#     # Get tokenizer
#     tokenizer_args = tokenizer_args or {}
#     if (tokenizer is None) and (not no_cache):
#         with _CACHE_LOCK:
#             if tokenizer_args != _TOKENIZER_CACHE_ARGS:
#                 # Args changed; drop cache
#                 _TOKENIZER_CACHE.pop(lang, None)

#             tokenizer = _TOKENIZER_CACHE.get(lang)

#     if tokenizer is None:
#         tokenizer = get_tokenizer(lang, **tokenizer_args)

#         if not no_cache:
#             with _CACHE_LOCK:
#                 _TOKENIZER_CACHE[lang] = tokenizer
#                 _TOKENIZER_CACHE_ARGS = tokenizer_args

#     # Get phonemizer
#     phonemizer_args = phonemizer_args or {}
#     if (phonemizer is None) and (not no_cache):
#         with _CACHE_LOCK:
#             if phonemizer_args != _PHONEMIZER_CACHE_ARGS:
#                 # Args changed; drop cache
#                 _PHONEMIZER_CACHE.pop(lang, None)

#             phonemizer = _PHONEMIZER_CACHE.get(lang)

#     if phonemizer is None:
#         phonemizer = get_phonemizer(lang, **phonemizer_args)

#         if not no_cache:
#             _PHONEMIZER_CACHE[lang] = phonemizer
#             _PHONEMIZER_CACHE_ARGS = tokenizer_args

#     # phonemize(**kwargs)
#     phonemize_kwargs = {}
#     if inline_pronunciations is not None:
#         phonemize_kwargs["inline_pronunciations"] = inline_pronunciations

#     sentences = tokenizer.tokenize(text)

#     if return_format == TextToPhonemesReturn.SENTENCES:
#         return_sentences: SENTENCES_TYPE = []
#         for sentence in sentences:
#             sentence.phonemes = list(
#                 phonemizer.phonemize(sentence.tokens, **phonemize_kwargs)
#             )
#             return_sentences.append(sentence)

#         return return_sentences

#     if return_format == TextToPhonemesReturn.WORD_PHONEMES:
#         return_word_phonemes: WORD_PHONEMES_TYPE = []

#         for sentence in sentences:
#             # Return phonemes grouped by word
#             return_word_phonemes.extend(
#                 phonemizer.phonemize(sentence.tokens, **phonemize_kwargs)
#             )

#         return return_word_phonemes

#     if return_format == TextToPhonemesReturn.SENTENCE_PHONEMES:
#         # Return phonemes grouped by sentence only
#         return_sentence_phonemes: SENTENCE_PHONEMES_TYPE = []

#         for sentence in sentences:
#             return_sentence_phonemes.append(
#                 [
#                     phoneme
#                     for word_phonemes in phonemizer.phonemize(
#                         sentence.tokens, **phonemize_kwargs
#                     )
#                     for phoneme in word_phonemes
#                 ]
#             )

#         return return_sentence_phonemes

#     if return_format == TextToPhonemesReturn.SENTENCE_WORD_PHONEMES:
#         # Return phonemes grouped by sentence and word
#         return [
#             list(phonemizer.phonemize(sentence.tokens, **phonemize_kwargs))
#             for sentence in sentences
#         ]

#     if return_format == TextToPhonemesReturn.FLAT_PHONEMES:
#         # Return flat list of phonemes
#         return_flat_phonemes: FLAT_PHONEMES_TYPE = []
#         for sentence in sentences:
#             return_flat_phonemes.extend(
#                 phoneme
#                 for word_phonemes in phonemizer.phonemize(
#                     sentence.tokens, **phonemize_kwargs
#                 )
#                 for phoneme in word_phonemes
#             )

#         return return_flat_phonemes

#     # if return_format == TextToPhonemesReturn.WORD_TUPLES:
#     # Return tuples of (sentence index, word text, word phonemes)
#     return_tuples: WORD_TUPLES_TYPE = []

#     for sentence_idx, sentence in enumerate(sentences):
#         return_tuples.extend(
#             (sentence_idx, token.text, token_phonemes)
#             for token, token_phonemes in zip(
#                 sentence.tokens,
#                 phonemizer.phonemize(sentence.tokens, **phonemize_kwargs),
#             )
#         )

#     return return_tuples


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
