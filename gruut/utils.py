"""Utility methods for gruut"""
import gzip
import itertools
import os
import re
import typing
from pathlib import Path

# Excludes 0xA0
_WHITESPACE = re.compile(r"[ \t]+")

# word -> [[p1, p2], [p1, p2, p3]]
LEXICON_TYPE = typing.Dict[
    str, typing.List[typing.Union[typing.List[str], typing.Tuple[str, ...]]]
]

# word(n) in lexicon
_WORD_WITH_NUMBER = re.compile(r"^([^(]+)(\(\d+\))$")


def load_lexicon(
    lexicon_file: typing.IO[str],
    word_separator: typing.Optional[str] = None,
    phoneme_separator: typing.Optional[str] = None,
    lexicon: typing.Optional[LEXICON_TYPE] = None,
    casing: typing.Optional[typing.Callable[[str], str]] = None,
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

    for line in lexicon_file:
        line = line.strip()
        if not line:
            continue

        word, phoneme_str = word_regex.split(line, maxsplit=1)
        phonemes = tuple(phoneme_regex.split(phoneme_str))

        word_match = _WORD_WITH_NUMBER.match(word)
        if word_match:
            # Strip (n) from word(n)
            word = word_match.group(1)

        if casing:
            # Apply case transformation
            word = casing(word)

        word_prons = lexicon.get(word)
        if word_prons:
            if phonemes not in word_prons:
                word_prons.append(phonemes)
        else:
            lexicon[word] = [phonemes]

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
