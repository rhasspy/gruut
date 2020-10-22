"""Class for getting phonetic pronunciations for cleaned text"""
import logging
import os
import re
import typing
from pathlib import Path

import phonetisaurus
import pydash

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

        # End of sentence symbols
        self.major_breaks: typing.Set[str] = set(
            pydash.get(self.config, "symbols.major_breaks", [])
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

    # -------------------------------------------------------------------------

    def phonemize(
        self,
        words: typing.List[str],
        word_indexes: bool = False,
        guess_word: typing.Optional[
            typing.Callable[[str], typing.Optional[typing.List[PRONUNCIATION_TYPE]]]
        ] = None,
        word_breaks: bool = False,
        minor_breaks: bool = True,
        major_breaks: bool = True,
    ) -> typing.List[typing.List[PRONUNCIATION_TYPE]]:
        """Get all possible pronunciations for cleaned words"""
        sentence_prons: typing.List[typing.List[PRONUNCIATION_TYPE]] = []
        missing_words: typing.List[typing.Tuple[int, str]] = []
        between_words = False

        if word_breaks:
            # Add initial word break
            sentence_prons.append([[IPA.BREAK_WORD.value]])

        for word in words:
            if word in self.minor_breaks:
                if word_breaks and between_words:
                    # Add end of word break
                    sentence_prons.append([[IPA.BREAK_WORD.value]])

                # Minor break (short pause)
                if minor_breaks:
                    sentence_prons.append([[IPA.BREAK_MINOR.value]])

                between_words = True
                continue

            if word in self.major_breaks:
                if word_breaks and between_words:
                    # Add end of word break
                    sentence_prons.append([[IPA.BREAK_WORD.value]])

                # Major break (sentence boundary)
                if major_breaks:
                    sentence_prons.append([[IPA.BREAK_MAJOR.value]])

                between_words = False
                continue

            # if word == "?":
            #     if self.question_mark and sentence_prons:
            #         # Add rising intonation to previous word
            #         prev_prons = sentence_prons[-1]
            #         for prev_idx, prev_pron in enumerate(prev_prons):
            #             prev_prons[prev_idx] = [IPA.INTONATION_RISING.value] + list(
            #                 prev_pron
            #             )

            #     # Assume major break
            #     sentence_prons.append([[IPA.BREAK_MAJOR.value]])
            #     between_words = False
            #     continue

            if word_breaks and between_words:
                # Add IPA word break symbol between words
                sentence_prons.append([[IPA.BREAK_WORD.value]])

            # Actual word
            between_words = True
            index: typing.Optional[int] = None
            if word_indexes:
                index_match = WORD_WITH_INDEX.match(word)
                if index_match:
                    word = index_match.group(1)
                    index = int(index_match.group(2))

            word_prons = self.lexicon.get(word)
            word_guessed = False

            if not word_prons and guess_word:
                # Use supplied function
                word_guessed = True
                word_prons = guess_word(word)
                if word_prons is not None:
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
                if not word_guessed:
                    # Need to guess
                    missing_words.append((len(sentence_prons), word))

                # Add placeholder
                sentence_prons.append([])

        words_to_guess = set(w for _, w in missing_words)

        if words_to_guess:
            _LOGGER.debug("Guessing pronunciations for %s", words_to_guess)
            for word, word_phonemes in self.predict(words=words_to_guess):
                # Add to lexicon
                self.lexicon[word] = [word_phonemes]

            # Fill in missing words
            for word_idx, word in missing_words:
                word_prons = self.lexicon.get(word)
                if word_prons:
                    sentence_prons[word_idx] = word_prons

        if between_words and word_breaks:
            # Add final word break
            sentence_prons.append([[IPA.BREAK_WORD.value]])

        return sentence_prons

    # -------------------------------------------------------------------------

    def predict(
        self, words: typing.Iterable[str], **kwargs
    ) -> typing.Iterable[typing.Tuple[str, typing.List[str]]]:
        """Predict word pronunciations using built-in g2p model"""
        for result in phonetisaurus.predict(
            words, model_path=self.g2p_model_path, **kwargs
        ):
            yield result
