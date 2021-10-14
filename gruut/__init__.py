"""gruut module"""
import itertools
import logging
import re
import sqlite3
import threading
import typing
from enum import Enum
from pathlib import Path

from gruut.const import KNOWN_LANGS, TextProcessorSettings
from gruut.text_processor import Sentence, TextProcessor
from gruut.utils import resolve_lang

# -----------------------------------------------------------------------------

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("gruut")

__version__ = (_DIR / "VERSION").read_text().strip()
__author__ = "Michael Hansen (synesthesiam)"
__all__ = [
    "sentences",
    "is_language_supported",
    "get_supported_languages",
    "TextProcessor",
    "TextProcessorSettings",
]

# -----------------------------------------------------------------------------

_LOCAL = threading.local()
_PROCESSORS_LOCK = threading.RLock()


def sentences(
    text: str,
    lang: str = "en_US",
    ssml: bool = False,
    espeak: bool = False,
    major_breaks: bool = True,
    minor_breaks: bool = True,
    punctuations: bool = True,
    explicit_lang: bool = True,
    phonemes: bool = True,
    break_phonemes: bool = True,
    pos: bool = True,
    **process_args,
) -> typing.Iterable[Sentence]:
    """
    Process text and return sentences

    Args:
        text: input text or SSML (ssml=True)
        lang: default language of input text
        ssml: True if input text is SSML
        espeak: True if eSpeak phonemes should be used
        major_breaks: False if no sentence-breaking symbols in output
        minor_breaks: False if no phrase-breaking symbols in output
        punctuations: False if no word-surrounding symbols in output
        **process_args: keyword arguments passed to TextProcessor.process

    Returns:
        sentences: iterable of Sentence objects

    """
    model_prefix = "" if (not espeak) else "espeak"

    with _PROCESSORS_LOCK:
        if not hasattr(_LOCAL, "processors"):
            _LOCAL.processors = {}

        text_processor = _LOCAL.processors.get(model_prefix)
        if text_processor is None:
            text_processor = TextProcessor(default_lang=lang, model_prefix=model_prefix)
            _LOCAL.processors[model_prefix] = text_processor

    assert text_processor is not None
    graph, root = text_processor(text, lang=lang, ssml=ssml, **process_args)

    yield from text_processor.sentences(
        graph,
        root,
        major_breaks=major_breaks,
        minor_breaks=minor_breaks,
        punctuations=punctuations,
        explicit_lang=explicit_lang,
        phonemes=phonemes,
        break_phonemes=break_phonemes,
        pos=pos,
    )


# -----------------------------------------------------------------------------


def is_language_supported(lang: str) -> bool:
    """True if gruut supports lang"""
    return resolve_lang(lang) in KNOWN_LANGS


def get_supported_languages() -> typing.Set[str]:
    """Set of supported gruut languages"""
    return set(KNOWN_LANGS)
