"""Utility methods for gruut"""
import base64
import itertools
import logging
import os
import re
import typing
from enum import Enum
from pathlib import Path

import gruut_ipa

from .const import REGEX_MATCH, REGEX_PATTERN

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("gruut.utils")

# -----------------------------------------------------------------------------


def find_lang_dir(
    lang: str,
    search_dirs: typing.Optional[typing.Iterable[typing.Union[str, Path]]] = None,
) -> typing.Optional[Path]:
    """
    Search for a language's model directory by name.

    Tries to find a directory by:

    #. Importing a module name ``gruut_lang_<short_lang>`` where short_lang is "en" for "en-us", etc.
    #. Looking for ``<lang>/lexicon.db`` in each directory in order:

       * ``search_dirs``
       * ``$XDG_CONFIG_HOME/gruut``
       * A "data" directory next to the gruut module
       * A "data" directory inside gruut module

    Args:
        lang: Full language name (e.g., en-us)
        search_dirs: Optional iterable of directory paths to search first

    Returns:
        Path to the language model directory or None if it can't be found
    """
    try:
        base_lang = lang.split("-")[0]
        lang_module_name = f"gruut_lang_{base_lang}"
        lang_module = __import__(lang_module_name)

        _LOGGER.debug("Successfully imported %s", lang_module_name)

        return lang_module.get_lang_dir()
    except ImportError:
        _LOGGER.debug("Tried to import module for %s", lang)
        pass

    search_dirs = typing.cast(typing.List[Path], [Path(p) for p in search_dirs or []])

    # ${XDG_CONFIG_HOME}/gruut or ${HOME}/gruut
    maybe_config_home = os.environ.get("XDG_CONFIG_HOME")
    if maybe_config_home:
        search_dirs.append(Path(maybe_config_home) / "gruut")
    else:
        search_dirs.append(Path.home() / ".config" / "gruut")

    # Data directory *next to* gruut
    search_dirs.append(_DIR.parent / "data")

    # Data directory *inside* gruut
    search_dirs.append(_DIR / "data")

    _LOGGER.debug("Searching %s for language file(s)", search_dirs)

    for check_dir in search_dirs:
        lang_dir = check_dir / lang
        lexicon_path = lang_dir / "lexicon.db"
        if lexicon_path.is_file():
            _LOGGER.debug("Found language file(s) in %s", lang_dir)
            return lang_dir

    return None


# -----------------------------------------------------------------------------


def maybe_compile_regex(
    str_or_pattern: typing.Union[str, REGEX_PATTERN]
) -> REGEX_PATTERN:
    """Compile regex pattern if it's a string"""
    if isinstance(str_or_pattern, REGEX_PATTERN):
        return str_or_pattern

    assert isinstance(str_or_pattern, str)

    return re.compile(str_or_pattern)


# -----------------------------------------------------------------------------


def get_currency_names(locale_str: str) -> typing.Dict[str, str]:
    """
    Try to get currency names and symbols for a Babel locale.

    Returns:
        Dictionary whose keys are currency symbols (like "$") and whose values are currency names (like "USD")
    """
    currency_names = {}

    try:
        import babel
        import babel.numbers

        locale = babel.Locale(locale_str)
        currency_names = {
            babel.numbers.get_currency_symbol(cn): cn for cn in locale.currency_symbols
        }
    except ImportError:
        # Expected if babel is not installed
        pass
    except Exception:
        _LOGGER.warning("get_currency_names")

    return currency_names


# -----------------------------------------------------------------------------

INLINE_PHONEMES_PATTERN = re.compile(r"\B\[\[([^\]]+)\]\]\B")
ENCODED_PHONEMES_PATTERN = re.compile(r"^__phonemes_([^_]+)__$")

INLINE_SOUNDSLIKE_PATTERN = re.compile(r"\B{{(.+)}}\B")
ENCODED_SOUNDSLIKE_PATTERN = re.compile(r"^__soundslike_([^_]+)__$")

SOUNDSLIKE_SEGMENTS_PATTERN = re.compile(r"({[^}]+)}")
SOUNDSLIKE_SEGMENTS_TRANSLATE = str.maketrans({"{": None, "}": None})


class InlinePronunciationType(str, Enum):
    """Types of inline pronunciations in text"""

    PHONEMES = "phonemes"
    """Whitespace-separated [[ p h o n e m e s ]]"""

    SOUNDS_LIKE = "sounds_like"
    """{{ Words with optional s{eg}m{ent}s }}"""


# Allow ' for primary stress and , for secondary stress
# Allow : for elongation
IPA_TRANSLATE = str.maketrans(
    "',:",
    "".join(
        [
            gruut_ipa.IPA.STRESS_PRIMARY.value,
            gruut_ipa.IPA.STRESS_SECONDARY.value,
            gruut_ipa.IPA.LONG,
        ]
    ),
)


def encode_inline_pronunciations(text: str, phonemes: gruut_ipa.Phonemes) -> str:
    """Encode inline phonemes in text using __phonemes_<base32-phonemes>__ format"""

    def replace_phonemes(match: REGEX_MATCH) -> str:
        ipa = match.group(1).strip()
        ipa = ipa.translate(IPA_TRANSLATE)

        # Normalize and separate with whitespace
        norm_ipa = " ".join(p.text for p in phonemes.split(ipa))

        # Base32 is used here because it's insensitive to case transformations
        b32_ipa = base64.b32encode(norm_ipa.encode()).decode()

        inline_key = f"__phonemes_{b32_ipa}__"

        return inline_key

    def replace_soundslike(match: REGEX_MATCH) -> str:
        words = match.group(1).strip()

        # Base32 is used here because it's insensitive to case transformations
        b32_words = base64.b32encode(words.encode()).decode()

        inline_key = f"__soundslike_{b32_words}__"

        return inline_key

    text = INLINE_PHONEMES_PATTERN.sub(replace_phonemes, text)
    text = INLINE_SOUNDSLIKE_PATTERN.sub(replace_soundslike, text)

    return text


def decode_inline_pronunciation(
    word: str,
) -> typing.Optional[typing.Tuple[InlinePronunciationType, str]]:
    """Return encoded inline phonemes from word encoded as __phonemes_<base32-phonemes>__"""
    match = ENCODED_PHONEMES_PATTERN.match(word)
    if match:
        phonemes = base64.b32decode(match.group(1).upper().encode()).decode()
        return (InlinePronunciationType.PHONEMES, phonemes)

    match = ENCODED_SOUNDSLIKE_PATTERN.match(word)
    if match:
        sounds_like = base64.b32decode(match.group(1).upper().encode()).decode()
        return (InlinePronunciationType.SOUNDS_LIKE, sounds_like)

    return None


def get_sounds_like_segments(word: str) -> typing.Iterable[typing.Tuple[int, int]]:
    """Get start/end indexes for s{eg}m{ent}s of words"""
    offset = 0
    for match in SOUNDSLIKE_SEGMENTS_PATTERN.finditer(word):
        yield (match.start(1) - offset, match.end(1) - offset - 1)
        offset += 2  # for { and }


def strip_sounds_like_segments(word: str) -> str:
    """Remove { } markers from sounds-like word"""
    return word.translate(SOUNDSLIKE_SEGMENTS_TRANSLATE)


# -----------------------------------------------------------------------------


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
