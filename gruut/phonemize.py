"""Class for getting phonetic pronunciations for cleaned text"""
import logging
import os
import re
import typing
from pathlib import Path

import pydash
from phonetisaurus import predict

from gruut_ipa import IPA

from .utils import LEXICON_TYPE, load_lexicon, maybe_gzip_open

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut.phonemize")

# List of phonemes for each word
PRONUNCIATION_TYPE = typing.Union[typing.List[str], typing.Tuple[str, ...]]


WORD_WITH_INDEX = re.compile(r"^([^_]+)_(\d+)$")

# -----------------------------------------------------------------------------


class Phonemizer:
    """Gets phonetic pronunciations for clean words"""

    def __init__(self, config, lexicon: typing.Optional[LEXICON_TYPE] = None):
        self.config = config

        # Short pause symbols (commas, etc.)
        self.minor_breaks: typing.Set[str] = set(
            pydash.get(self.config, "symbols.minor_breaks", [])
        )

        # End of sentence symbol
        self.major_break: typing.Optional[str] = pydash.get(
            self.config, "symbols.major_break"
        )

        # If True, question marks add rising intonation to the previous word
        self.question_mark = bool(
            pydash.get(self.config, "symbols.question_mark", False)
        )

        # Case transformation (lower/upper)
        casing = pydash.get(self.config, "symbols.casing")
        self.casing: typing.Optional[typing.Callable[[str], str]] = None
        if casing == "lower":
            self.casing = str.lower
        elif casing == "upper":
            self.casing = str.upper

        self.g2p_model_path = Path(pydash.get(self.config, "g2p.model"))

        self.lexicon: LEXICON_TYPE = {}
        if lexicon:
            self.lexicon = lexicon
        else:
            # Load lexicons
            for lexicon_path in self.config.get("lexicons", []):
                if os.path.isfile(lexicon_path):
                    _LOGGER.debug("Loading lexicon from %s", lexicon_path)
                    with maybe_gzip_open(lexicon_path, "r") as lexicon_file:
                        load_lexicon(
                            lexicon_file, lexicon=self.lexicon, casing=self.casing
                        )
                else:
                    _LOGGER.warning("Skipping lexicon at %s", lexicon_path)

            _LOGGER.debug("Loaded pronunciations for %s word(s)", len(self.lexicon))

    def phonemize(
        self,
        words: typing.List[str],
        word_indexes: bool = False,
        guess_word: typing.Optional[
            typing.Callable[[str], typing.Optional[typing.List[PRONUNCIATION_TYPE]]]
        ] = None,
    ) -> typing.List[typing.List[PRONUNCIATION_TYPE]]:
        """Get all possible pronunciations for cleaned words"""
        sentence_prons: typing.List[typing.List[PRONUNCIATION_TYPE]] = []
        missing_words: typing.List[typing.Tuple[int, str]] = []

        for word_idx, word in enumerate(words):
            if word in self.minor_breaks:
                # Minor break (short pause)
                sentence_prons.append([[IPA.BREAK_MINOR.value]])
                continue

            if word == self.major_break:
                # Major break (sentence boundary)
                sentence_prons.append([[IPA.BREAK_MAJOR.value]])
                continue

            if word == "?":
                if self.question_mark and word_idx > 0:
                    # Add rising intonation to previous word
                    prev_prons = sentence_prons[word_idx - 1]
                    for prev_idx, prev_pron in enumerate(prev_prons):
                        prev_prons[prev_idx] = [IPA.INTONATION_RISING.value] + list(
                            prev_pron
                        )

                # Skip question mark
                continue

            index: typing.Optional[int] = None
            if word_indexes:
                index_match = WORD_WITH_INDEX.match(word)
                if index_match:
                    word = index_match.group(1)
                    index = int(index_match.group(2))

            word_prons = self.lexicon.get(word)

            if not word_prons and guess_word:
                # Use supplied function
                word_prons = guess_word(word)
                if word_prons:
                    # Update lexicon
                    self.lexicon[word] = word_prons

            if word_prons:
                # In lexicon
                if index is None:
                    # All pronunciations
                    sentence_prons.append(word_prons)
                else:
                    # Specific pronunciation.
                    # Clamp 1-based index.
                    pron_index = max(0, index - 1) % len(word_prons)
                    sentence_prons.append([word_prons[pron_index]])
            else:
                # Need to guess
                sentence_prons.append([])
                missing_words.append((word_idx, word))

        words_to_guess = set(w for _, w in missing_words)

        if words_to_guess:
            _LOGGER.debug("Guessing pronunciations for %s", words_to_guess)
            for word, word_phonemes in predict(
                words=words_to_guess, model_path=self.g2p_model_path
            ):
                # Add to lexicon
                self.lexicon[word] = [word_phonemes]

            # Fill in missing words
            for word_idx, word in missing_words:
                word_prons = self.lexicon.get(word)
                if word_prons:
                    sentence_prons[word_idx] = word_prons

        return sentence_prons
