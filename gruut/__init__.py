"""gruut module"""
import threading
import typing
from enum import Enum
from pathlib import Path

from .const import WORD_PHONEMES, Sentence, Token, TokenFeatures, WordPronunciation
from .lang import KNOWN_LANGS, get_phonemizer, get_tokenizer, resolve_lang
from .phonemize import Phonemizer, SqlitePhonemizer, UnknownWordError
from .toksen import RegexTokenizer, Tokenizer

_DIR = Path(__file__).parent

__version__ = (_DIR / "VERSION").read_text().strip()

# -----------------------------------------------------------------------------

_TOKENIZER_CACHE: typing.Dict[str, Tokenizer] = {}
_TOKENIZER_CACHE_ARGS: typing.Mapping[str, typing.Any] = {}
_TOKENIZER_CACHE_LOCK = threading.RLock()

_PHONEMIZER_CACHE: typing.Dict[str, Phonemizer] = {}
_PHONEMIZER_CACHE_ARGS: typing.Mapping[str, typing.Any] = {}
_PHONEMIZER_CACHE_LOCK = threading.RLock()

# -----------------------------------------------------------------------------


class TextToPhonemesReturn(str, Enum):
    """
    Format of return value from :py:meth:`~gruut.text_to_phonemes`.
    """

    WORD_TUPLES = "word_tuples"
    """Tuples of the form (sentence index, word, word phonemes)"""

    FLAT_PHONEMES = "flat_phonemes"
    """Flat list of phoneme strings"""

    WORD_PHONEMES = "word_phonemes"
    """Lists of phonemes grouped by words only"""

    SENTENCE_PHONEMES = "sentence_phonemes"
    """Lists of phonemes grouped by sentence only"""

    SENTENCE_WORD_PHONEMES = "sentence_word_phonemes"
    """Lists of phonemes grouped by sentence, then word"""

    SENTENCES = "sentences"
    """List of :py:class:`~gruut.const.Sentence` objects with tokens, features, etc."""


# (sentence index, word, [phonemes])
WORD_TUPLES_TYPE = typing.List[typing.Tuple[int, str, WORD_PHONEMES]]
FLAT_PHONEMES_TYPE = typing.List[str]
WORD_PHONEMES_TYPE = typing.List[WORD_PHONEMES]
SENTENCE_PHONEMES_TYPE = typing.List[WORD_PHONEMES]
SENTENCE_WORD_PHONEMES_TYPE = typing.List[typing.List[WORD_PHONEMES]]
SENTENCES_TYPE = typing.List[Sentence]


TEXT_TO_PHONEMES_RETURN_TYPE = typing.Union[
    WORD_TUPLES_TYPE,
    FLAT_PHONEMES_TYPE,
    WORD_PHONEMES_TYPE,
    SENTENCE_PHONEMES_TYPE,
    SENTENCE_WORD_PHONEMES_TYPE,
    SENTENCES_TYPE,
]

# -----------------------------------------------------------------------------


def text_to_phonemes(
    text: str,
    lang: str = "en-us",
    return_format: typing.Union[str, TextToPhonemesReturn] = "word_tuples",
    no_cache: bool = False,
    tokenizer: typing.Optional[Tokenizer] = None,
    tokenizer_args: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    phonemizer: typing.Optional[Phonemizer] = None,
    phonemizer_args: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> TEXT_TO_PHONEMES_RETURN_TYPE:
    """
    High-level text to phonemes interface.

    Args:
        text: Text to tokenize and phonemize
        lang: Language of the text
        return_format: Format of return value
        no_cache: If True, tokenizer/phonemizer cache are not used
        tokenizer: Optional tokenizer to use instead of creating one
        tokenizer_args: Optional keyword arguments used when creating tokenizer
        phonemizer: Optional phonemizer to use instead of creating one
        phonemizer_args: Optional keyword arguments used when creating phonemizer

    Returns:
        Tuples of the form (sentence index, word, [word phonemes]). See TextToPhonemesReturn enum for more return formats.

    Example:
        ::

            from gruut import text_to_phonemes

            text = 'He wound it around the wound, saying "I read it was $10 to read."'

            for sent_idx, word, word_phonemes in text_to_phonemes(text, lang="en-us"):
                print(word, *word_phonemes)

        Output::

            he h ˈi
            wound w ˈaʊ n d
            it ˈɪ t
            around ɚ ˈaʊ n d
            the ð ə
            wound w ˈu n d
            , |
            saying s ˈeɪ ɪ ŋ
            i ˈaɪ
            read ɹ ˈɛ d
            it ˈɪ t
            was w ə z
            ten t ˈɛ n
            dollars d ˈɑ l ɚ z
            to t ə
            read ɹ ˈi d
            . ‖
    """
    global _TOKENIZER_CACHE_ARGS, _PHONEMIZER_CACHE_ARGS

    lang = resolve_lang(lang)

    # Get tokenizer
    tokenizer_args = tokenizer_args or {}
    if (tokenizer is None) and (not no_cache):
        with _TOKENIZER_CACHE_LOCK:
            if tokenizer_args != _TOKENIZER_CACHE_ARGS:
                # Args changed; drop cache
                _TOKENIZER_CACHE.pop(lang, None)

            tokenizer = _TOKENIZER_CACHE.get(lang)

    if tokenizer is None:
        tokenizer = get_tokenizer(lang, **tokenizer_args)

        if not no_cache:
            with _TOKENIZER_CACHE_LOCK:
                _TOKENIZER_CACHE[lang] = tokenizer
                _TOKENIZER_CACHE_ARGS = tokenizer_args

    # Get phonemizer
    phonemizer_args = phonemizer_args or {}
    if (phonemizer is None) and (not no_cache):
        with _PHONEMIZER_CACHE_LOCK:
            if phonemizer_args != _PHONEMIZER_CACHE_ARGS:
                # Args changed; drop cache
                _PHONEMIZER_CACHE.pop(lang, None)

            phonemizer = _PHONEMIZER_CACHE.get(lang)

    if phonemizer is None:
        phonemizer = get_phonemizer(lang, **phonemizer_args)

        if not no_cache:
            _PHONEMIZER_CACHE[lang] = phonemizer
            _PHONEMIZER_CACHE_ARGS = tokenizer_args

    sentences = tokenizer.tokenize(text)

    if return_format == TextToPhonemesReturn.SENTENCES:
        return_sentences: SENTENCES_TYPE = []
        for sentence in sentences:
            sentence.phonemes = list(phonemizer.phonemize(sentence.tokens))
            return_sentences.append(sentence)

        return return_sentences

    if return_format == TextToPhonemesReturn.WORD_PHONEMES:
        return_word_phonemes: WORD_PHONEMES_TYPE = []

        for sentence in sentences:
            # Return phonemes grouped by word
            return_word_phonemes.extend(phonemizer.phonemize(sentence.tokens))

        return return_word_phonemes

    if return_format == TextToPhonemesReturn.SENTENCE_PHONEMES:
        # Return phonemes grouped by sentence only
        return_sentence_phonemes: SENTENCE_PHONEMES_TYPE = []

        for sentence in sentences:
            return_sentence_phonemes.append(
                [
                    phoneme
                    for word_phonemes in phonemizer.phonemize(sentence.tokens)
                    for phoneme in word_phonemes
                ]
            )

        return return_sentence_phonemes

    if return_format == TextToPhonemesReturn.SENTENCE_WORD_PHONEMES:
        # Return phonemes grouped by sentence and word
        return [list(phonemizer.phonemize(sentence.tokens)) for sentence in sentences]

    if return_format == TextToPhonemesReturn.FLAT_PHONEMES:
        # Return flat list of phonemes
        return_flat_phonemes: FLAT_PHONEMES_TYPE = []
        for sentence in sentences:
            return_flat_phonemes.extend(
                phoneme
                for word_phonemes in phonemizer.phonemize(sentence.tokens)
                for phoneme in word_phonemes
            )

        return return_flat_phonemes

    # if return_format == TextToPhonemesReturn.WORD_TUPLES:
    # Return tuples of (sentence index, word text, word phonemes)
    return_tuples: WORD_TUPLES_TYPE = []

    for sentence_idx, sentence in enumerate(sentences):
        return_tuples.extend(
            (sentence_idx, word, word_phonemes)
            for word, word_phonemes in zip(
                sentence.clean_words, phonemizer.phonemize(sentence.tokens)
            )
        )

    return return_tuples


# -----------------------------------------------------------------------------


def is_language_supported(lang: str) -> bool:
    """True if gruut supports lang"""
    return resolve_lang(lang) in KNOWN_LANGS


def get_supported_languages() -> typing.Set[str]:
    """Set of supported gruut languages"""
    return set(KNOWN_LANGS)
