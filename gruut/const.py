"""Shared classes, types, and enums"""
import re
import typing
from dataclasses import dataclass, field
from enum import Enum

REGEX_TYPE = typing.Union[str, re.Pattern]

WORD_PHONEMES = typing.Sequence[str]


class TokenFeatures(str, Enum):
    """Commonly used token features"""

    PART_OF_SPEECH = "pos"


@dataclass
class Token:
    """Single token"""

    text: str

    # Optional token features (e.g., part of speech)
    features: typing.MutableMapping[str, str] = field(default_factory=dict)


TOKEN_OR_STR = typing.Union[Token, str]


@dataclass
class WordPronunciation:
    """Single pronunciation for a word"""

    phonemes: typing.Sequence[str]

    # Features that cause this word pronunciation to be preferred
    preferred_features: typing.MutableMapping[str, typing.Set[str]] = field(
        default_factory=dict
    )


@dataclass
class Sentence:
    """Tokenized and cleaned sentence"""

    # Original text
    raw_text: str = ""
    raw_words: typing.Sequence[str] = field(default_factory=list)

    clean_text: str = ""
    clean_words: typing.Sequence[str] = field(default_factory=list)
    tokens: typing.Sequence[Token] = field(default_factory=list)
