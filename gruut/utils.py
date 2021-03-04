"""Utility methods for gruut"""
import gzip
import itertools
import logging
import os
import re
import typing
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WordPronunciation:
    """Single pronunciation for a word"""

    phonemes: typing.Union[typing.List[str], typing.Tuple[str, ...]]
    valid_pos: typing.Optional[typing.Set[str]] = None


# Excludes 0xA0
_WHITESPACE = re.compile(r"[ \t]+")

# word -> [[p1, p2], [p1, p2, p3]]
LEXICON_TYPE = typing.Dict[str, typing.List[WordPronunciation]]

# word(n) in lexicon
_WORD_WITH_NUMBER = re.compile(r"^([^(]+)(\(\d+\))$")

_LOGGER = logging.getLogger("gruut.utils")


def load_lexicon(
    lexicon_file: typing.IO[str],
    word_separator: typing.Optional[str] = None,
    phoneme_separator: typing.Optional[str] = None,
    lexicon: typing.Optional[LEXICON_TYPE] = None,
    casing: typing.Optional[typing.Callable[[str], str]] = None,
    multi_word: bool = False,
) -> LEXICON_TYPE:
    """Load a CMU-style lexicon."""
    if lexicon is None:
        lexicon = {}

    if word_separator:
        word_regex = re.compile(word_separator)
    else:
        word_regex = _WHITESPACE

    if phoneme_separator:
        phoneme_regex = re.compile(phoneme_separator)
    else:
        phoneme_regex = _WHITESPACE

    for line_index, line in enumerate(lexicon_file):
        line = line.strip()
        if not line:
            continue

        try:
            if multi_word:
                # Possibly multiple words and phonemes, separated by whitespace
                word, phonemes_str = line.split("/", maxsplit=1)
                words = [w.replace(",", "") for w in _WHITESPACE.split(word)]
                assert (
                    "," not in phonemes_str
                ), "Cannot handle multiple pronunciations for multi-words"

                # Remove /separators/
                phonemes_str = phonemes_str.replace("/", "")
                word_phonemes_strs = [
                    [p.strip()] for p in _WHITESPACE.split(phonemes_str)
                ]
            else:
                # One word, one or more pronunciations
                word, phonemes_str = word_regex.split(line, maxsplit=1)
                words = [word]

                # Remove /separators/
                phonemes_str = phonemes_str.replace("/", "")

                word_phonemes_strs = [[p.strip() for p in phonemes_str.split(",")]]

            for word, phoneme_strs in zip(words, word_phonemes_strs):
                word_match = _WORD_WITH_NUMBER.match(word)
                if word_match:
                    # Strip (n) from word(n)
                    word = word_match.group(1)

                if casing:
                    # Apply case transformation
                    word = casing(word)

                # Multiple pronunciations separated by commas
                for phoneme_str in phoneme_strs:
                    if not phoneme_str:
                        continue

                    phonemes = tuple(phoneme_regex.split(phoneme_str))
                    word_pron = WordPronunciation(phonemes=phonemes)

                    word_prons = lexicon.get(word)
                    if word_prons:
                        word_prons.append(word_pron)
                    else:
                        lexicon[word] = [word_pron]
        except Exception as e:
            _LOGGER.exception("Error on line %s: %s", line_index + 1, line)
            raise e

    return lexicon


# -----------------------------------------------------------------------------


def maybe_gzip_open(
    path_or_str: typing.Union[Path, str], mode: str = "r", create_dir: bool = True
) -> typing.IO[typing.Any]:
    """Opens a file as gzip if it has a .gz extension."""
    if create_dir and mode in {"w", "a"}:
        Path(path_or_str).parent.mkdir(parents=True, exist_ok=True)

    if str(path_or_str).endswith(".gz"):
        if mode == "r":
            gzip_mode = "rt"
        elif mode == "w":
            gzip_mode = "wt"
        elif mode == "a":
            gzip_mode = "at"
        else:
            gzip_mode = mode

        return gzip.open(path_or_str, gzip_mode)

    return open(path_or_str, mode)


# -----------------------------------------------------------------------------


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


# -----------------------------------------------------------------------------


def env_constructor(loader, node):
    """Expand !env STRING to replace environment variables in STRING."""
    return os.path.expandvars(node.value)
