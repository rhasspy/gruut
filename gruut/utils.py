"""Utility methods for gruut"""
import gzip
import itertools
import logging
import os
import re
import typing
from dataclasses import dataclass
from pathlib import Path

import gruut_ipa

PHONEMES_TYPE = typing.Union[typing.List[str], typing.Tuple[str, ...]]


@dataclass
class Token:
    """Single token"""

    text: str
    pos: typing.Optional[str] = None  # part of speech


@dataclass
class WordPronunciation:
    """Single pronunciation for a word"""

    phonemes: PHONEMES_TYPE
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


# -----------------------------------------------------------------------------


def fa_word_pronunciation(word_pron, token):
    """Append e̞ in genitive case"""
    if token.pos == "Ne":
        word_pron = list(word_pron)
        word_pron.append("e̞")

    return word_pron


# -----------------------------------------------------------------------------

_WORD_BREAK = gruut_ipa.IPA.BREAK_WORD.value


def _fr_has_silent_consonant(last_char: str, last_phoneme: str) -> bool:
    """True if last consonant is silent in French"""
    # Credit: https://github.com/Remiphilius/PoemesProfonds/blob/master/lecture.py

    if last_char in {"d", "p", "t"}:
        return last_phoneme != last_char
    if last_char == "r":
        return last_phoneme != "ʁ"
    if last_char in {"s", "x", "z"}:
        return last_phoneme not in {"s", "z"}
    if last_char == "n":
        return last_phoneme not in {"n", "ŋ"}

    return False


def _fr_is_vowel(phoneme: str) -> bool:
    """True if phoneme is a French vowel"""
    return phoneme in {
        "i",
        "y",
        "u",
        "e",
        "ø",
        "o",
        "ə",
        "ɛ",
        "œ",
        "ɔ",
        "a",
        "ɔ̃",
        "ɛ̃",
        "ɑ̃",
        "œ̃",
    }


def fr_liason(
    tokens: typing.Iterable[Token],
    token_phonemes: typing.Iterable[PHONEMES_TYPE],
    word_breaks: bool = False,
):
    """Add liasons to a sentence by examining word texts, parts of speech, and phonemes."""
    if word_breaks:
        # Exclude word breaks for now
        token_phonemes = [p for p in token_phonemes if p != [_WORD_BREAK]]

        # Produce initial word break
        yield [_WORD_BREAK]

    for (token1, token1_pron), (token2, token2_pron) in pairwise(
        zip(itertools.chain(tokens, [None]), itertools.chain(token_phonemes, [None]))
    ):
        if token2 is None:
            # Final token
            yield token1_pron
            continue

        liason = False

        # Conditions to meet for liason check:
        # 1) token 1 ends with a silent consonant
        # 2) token 2 starts with a vowel (phoneme)

        last_char1 = token1.text[-1]
        ends_silent_consonant = _fr_has_silent_consonant(last_char1, token1_pron[-1])
        starts_vowel = _fr_is_vowel(token2_pron[0])

        if ends_silent_consonant and starts_vowel:
            # Handle mandatory liason cases
            # https://www.commeunefrancaise.com/blog/la-liaison

            if token1.text == "et":
                # No liason
                pass
            elif token1.pos in {"DET", "NUM"}:
                # Determiner/adjective -> noun
                liason = True
            elif (token1.pos == "PRON") and (token2.pos in {"AUX", "VERB"}):
                # Pronoun -> verb
                liason = True
            elif (token1.pos == "ADP") or (token1.text == "très"):
                # Preposition
                liason = True
            elif (token1.pos == "ADJ") and (token2.pos in {"NOUN", "PROPN"}):
                # Adjective -> noun
                liason = True
            elif token1.pos in {"AUX", "VERB"}:
                # Verb -> vowel
                liason = True

        if liason:
            # Apply liason
            # s -> z
            # p -> p
            # d|t -> d
            liason_pron = token1_pron

            if last_char1 in {"s", "x", "z"}:
                liason_pron.append("z")
            elif last_char1 == "d":
                liason_pron.append("t")
            elif last_char1 in {"t", "p", "n"}:
                # Final phoneme is same as char
                liason_pron.append(last_char1)

            yield liason_pron

            # (keep word break)
            if word_breaks:
                yield [_WORD_BREAK]
        else:
            # Keep pronunciations the same
            yield token1_pron

            if word_breaks:
                yield [_WORD_BREAK]
