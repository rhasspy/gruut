"""Shared classes, types, and enums"""
import re
import typing
from dataclasses import dataclass, field
from enum import Enum

try:
    # Python >= 3.7
    REGEX_PATTERN = re.Pattern  # type: ignore
    REGEX_MATCH = re.Match  # type: ignore
except AttributeError:
    # Python 3.6
    REGEX_PATTERN = typing.re.Pattern  # type: ignore
    REGEX_MATCH = typing.re.Match  # type: ignore

REGEX_TYPE = typing.Union[str, REGEX_PATTERN]

WORD_PHONEMES = typing.Sequence[str]


class TokenFeatures(str, Enum):
    """Commonly used token features"""

    PART_OF_SPEECH = "pos"
    """Part of speech tag, typically chosen from Universal Dependencies"""


@dataclass
class Token:
    """Single token"""

    text: str
    """Text value of token"""

    features: typing.MutableMapping[str, str] = field(default_factory=dict)
    """
    Optional token features (e.g., part of speech).

    Keys are feature names, such as those in :py:class:`~gruut.const.TokenFeatures`.

    Values are assigned by models like those in :py:mod:`gruut.pos` or :py:mod:`gruut.g2p`.
    """


TOKEN_OR_STR = typing.Union[Token, str]


@dataclass
class WordPronunciation:
    """Single pronunciation for a word"""

    phonemes: typing.Sequence[str]
    """List of phonemes for this pronunciation"""

    preferred_features: typing.MutableMapping[str, typing.Set[str]] = field(
        default_factory=dict
    )
    """
    Features that cause this word pronunciation to be preferred.

    Keys are feature names, such as those in :py:class:`~gruut.const.TokenFeatures`.

    Values are sets of compatible feature values, for example part of speech tags.
    When choosing among several pronunciations for a :py:class:`~gruut.const.Token`, the one with more compatible features will be chosen.
    """


@dataclass
class Sentence:
    """Tokenized and cleaned sentence"""

    raw_text: str = ""
    """Original text before any processing"""

    raw_words: typing.Sequence[str] = field(default_factory=list)
    """Words from original text before any processing"""

    clean_text: str = ""
    """Text after being processed (i.e., abbreviations/numbers expanded)"""

    clean_words: typing.Sequence[str] = field(default_factory=list)
    """Words after being processed. Derived from :py:attr:`tokens`."""

    tokens: typing.Sequence[Token] = field(default_factory=list)
    """Tokens with optional features (e.g., part of speech)"""

    phonemes: typing.Optional[typing.Sequence[WORD_PHONEMES]] = None
    """Optional phonetic pronunciations for :py:attr:`tokens`. Added by phonemizer."""
