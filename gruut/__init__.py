"""gruut module"""
import threading
import typing
from pathlib import Path

from .const import WORD_PHONEMES, Sentence, Token, TokenFeatures, WordPronunciation
from .lang import get_phonemizer, get_tokenizer, resolve_lang
from .phonemize import Phonemizer, SqlitePhonemizer, UnknownWordError
from .toksen import RegexTokenizer, Tokenizer

_DIR = Path(__file__).parent

__version__ = (_DIR / "VERSION").read_text().strip()

# -----------------------------------------------------------------------------

_TOKENIZER_CACHE: typing.Dict[str, Tokenizer] = {}
_TOKENIZER_CACHE_LOCK = threading.RLock()

_PHONEMIZER_CACHE: typing.Dict[str, Phonemizer] = {}
_PHONEMIZER_CACHE_LOCK = threading.RLock()

# -----------------------------------------------------------------------------


def text_to_phonemes(
    text: str,
    lang: str = "en-us",
    parallel: bool = False,
    return_sentences: bool = False,
    no_cache: bool = False,
    tokenizer: typing.Optional[Tokenizer] = None,
    phonemizer: typing.Optional[Phonemizer] = None,
    tokenizer_args: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    phonemizer_args: typing.Optional[typing.Mapping[str, typing.Any]] = None,
) -> typing.Sequence[typing.Union[Sentence, typing.Sequence[WORD_PHONEMES]]]:
    """
    High-level text to phonemes interface.

    Attributes
    ----------
    text: str
        Text to tokenize and phonemize.

    lang: str
        Language of the text.
        default: en-us

    return_sentences: bool
        If True, Sentence objects are returned instead of phoneme lists.
        default: False

    Returns
    -------
    phonemes: Sequence[Sequence[Sequence[str]]]
        List of phonemes for each word in each sentence of the text.
        sentences[words[phonemes[phoneme]]]

    If return_sentences is True, returns list of Sentence objects with phonemes
    attribute set.
    """
    lang = resolve_lang(lang)

    # Get tokenizer
    if (tokenizer is None) and (not no_cache):
        with _TOKENIZER_CACHE_LOCK:
            tokenizer = _TOKENIZER_CACHE.get(lang)

    if tokenizer is None:
        tokenizer_args = tokenizer_args or {}
        tokenizer = get_tokenizer(lang, **tokenizer_args)

        if not no_cache:
            with _TOKENIZER_CACHE_LOCK:
                _TOKENIZER_CACHE[lang] = tokenizer

    # Get phonemizer
    if (phonemizer is None) and (not no_cache):
        with _PHONEMIZER_CACHE_LOCK:
            phonemizer = _PHONEMIZER_CACHE.get(lang)

    if phonemizer is None:
        phonemizer_args = phonemizer_args or {}
        phonemizer = get_phonemizer(lang, **phonemizer_args)

        if not no_cache:
            _PHONEMIZER_CACHE[lang] = phonemizer

    if parallel:
        # Pre-load all pronunciations to avoid accessing the database
        with _PHONEMIZER_CACHE_LOCK:
            phonemizer.preload_prons()

    if return_sentences:
        # Return Sentence objects with all information
        all_sentences = []
        for sentence in tokenizer.tokenize(text):
            sentence.phonemes = list(phonemizer.phonemize(sentence.tokens))
            all_sentences.append(sentence)

        return all_sentences

    # Return phonemes only
    all_phonemes = []
    for sentence in tokenizer.tokenize(text):
        sentence_phonemes = list(phonemizer.phonemize(sentence.tokens))
        all_phonemes.append(sentence_phonemes)

    return all_phonemes
