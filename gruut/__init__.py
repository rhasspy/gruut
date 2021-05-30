"""gruut module"""
import threading
import typing

from .const import WORD_PHONEMES, Sentence, Token, TokenFeatures, WordPronunciation
from .lang import get_phonemizer, get_tokenizer, resolve_lang
from .phonemize import Phonemizer, SqlitePhonemizer, UnknownWordError
from .toksen import RegexTokenizer, Tokenizer

# -----------------------------------------------------------------------------

_TOKENIZER_CACHE: typing.Dict[str, Tokenizer] = {}
_TOKENIZER_CACHE_LOCK = threading.Lock()

_PHONEMIZER_CACHE: typing.Dict[str, Phonemizer] = {}
_PHONEMIZER_CACHE_LOCK = threading.Lock()

# -----------------------------------------------------------------------------


def text_to_phonemes(
    text: str,
    lang: str = "en-us",
    return_sentences: bool = False,
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
    with _TOKENIZER_CACHE_LOCK:
        tokenizer = _TOKENIZER_CACHE.get(lang)
        if tokenizer is None:
            tokenizer_args = tokenizer_args or {}
            tokenizer = get_tokenizer(lang, **tokenizer_args)
            _TOKENIZER_CACHE[lang] = tokenizer

    # Get phonemizer
    with _PHONEMIZER_CACHE_LOCK:
        phonemizer = _PHONEMIZER_CACHE.get(lang)
        if phonemizer is None:
            phonemizer_args = phonemizer_args or {}
            phonemizer = get_phonemizer(lang, **phonemizer_args)
            _PHONEMIZER_CACHE[lang] = phonemizer

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
