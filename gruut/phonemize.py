"""Class for getting phonetic pronunciations for cleaned text"""
import gzip
import logging
import os
import re
import shutil
import threading
import typing
import unicodedata
from pathlib import Path

import pydash

import phonetisaurus
from gruut_ipa import IPA

from .toksen import Token
from .utils import LEXICON_TYPE, load_lexicon, maybe_gzip_open

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut.phonemize")

# List of phonemes for each word
PRONUNCIATION_TYPE = typing.Union[typing.List[str], typing.Tuple[str, ...]]


WORD_WITH_INDEX = re.compile(r"^([^_]+)_(\d+)$")

NON_WORD_CHARS = re.compile(r"\W")

# -----------------------------------------------------------------------------


class Phonemizer:
    """Gets phonetic pronunciations for clean words"""

    def __init__(
        self,
        config,
        lexicon: typing.Optional[LEXICON_TYPE] = None,
        preload_lexicon: bool = True,
    ):
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

        self.has_tones = bool(pydash.get(self.config, "language.tones", []))

        # Case transformation (lower/upper)
        casing = pydash.get(self.config, "symbols.casing")
        self.casing: typing.Optional[typing.Callable[[str], str]] = None
        if casing == "lower":
            self.casing = str.lower
        elif casing == "upper":
            self.casing = str.upper

        self.g2p_model_path = Path(pydash.get(self.config, "g2p.model"))
        self.g2p_lock = threading.RLock()

        self.lexicon: LEXICON_TYPE = {}
        self.lexicon_loaded = False

        if lexicon:
            self.lexicon = lexicon
            self.lexicon_loaded = True
        elif preload_lexicon:
            # Load lexicons
            self.load_lexicon()
            _LOGGER.debug("Loaded pronunciations for %s word(s)", len(self.lexicon))

        self.g2p_lock = threading.RLock()

    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------

    def phonemize(
        self,
        tokens: typing.Sequence[typing.Union[str, Token]],
        word_indexes: bool = False,
        guess_word: typing.Optional[
            typing.Callable[[Token], typing.Optional[typing.List[PRONUNCIATION_TYPE]]]
        ] = None,
        word_breaks: bool = False,
        minor_breaks: bool = True,
        major_breaks: bool = True,
        separate_tones: typing.Optional[bool] = False,
        guess_with_word_chars: bool = True,
        process_pronunciation: typing.Optional[
            typing.Callable[[PRONUNCIATION_TYPE, Token], PRONUNCIATION_TYPE]
        ] = None,
    ) -> typing.List[typing.List[PRONUNCIATION_TYPE]]:
        """Get all possible pronunciations for cleaned words"""
        if not self.lexicon_loaded:
            # Dynamically load lexicon(s)
            self.load_lexicon()

        sentence_prons: typing.List[typing.List[PRONUNCIATION_TYPE]] = []
        missing_words: typing.List[typing.Tuple[int, Token]] = []
        between_words = False

        if separate_tones is None:
            # Separate only if language has tones
            separate_tones = self.has_tones

        if word_breaks:
            # Add initial word break
            sentence_prons.append([[IPA.BREAK_WORD.value]])

        for token_or_str in tokens:
            if isinstance(token_or_str, Token):
                token = token_or_str
            else:
                token = Token(text=token_or_str)

            word = token.text

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

            if not word_prons and guess_with_word_chars:
                # Try again with non-word characters removed
                filtered_word = Phonemizer.remove_nonword_chars(word)
                word_prons = self.lexicon.get(filtered_word)

            if not word_prons and guess_word:
                # Use supplied function
                word_guessed = True
                word_prons = guess_word(token)
                if word_prons is not None:
                    # Update lexicon
                    self.lexicon[word] = [
                        Phonemizer.maybe_separate_tones(wp, separate_tones)
                        for wp in word_prons
                    ]

            if word_prons:
                # Language-specific processing
                if process_pronunciation:
                    word_prons = [process_pronunciation(wp, token) for wp in word_prons]

                # In lexicon
                if index is None:
                    # All pronunciations
                    sentence_prons.append(
                        [
                            Phonemizer.maybe_separate_tones(wp, separate_tones)
                            for wp in word_prons
                        ]
                    )
                else:
                    # Specific pronunciation.
                    # Clamp 1-based index.
                    pron_index = max(0, index - 1) % len(word_prons)
                    sentence_prons.append(
                        [
                            Phonemizer.maybe_separate_tones(
                                word_prons[pron_index], separate_tones
                            )
                        ]
                    )
            else:
                if not word_guessed and self.is_word(token.text):
                    # Need to guess
                    missing_words.append((len(sentence_prons), token))

                # Add placeholder
                sentence_prons.append([])

        words_to_guess = set(t.text for _, t in missing_words)

        if words_to_guess:
            _LOGGER.debug("Guessing pronunciations for %s", words_to_guess)
            for word, word_phonemes in self.predict(words=words_to_guess):
                # Add to lexicon
                self.lexicon[word] = [word_phonemes]

            # Fill in missing words
            for word_idx, token in missing_words:
                word_prons = self.lexicon.get(token.text)
                if word_prons:
                    # Language-specific processing
                    if process_pronunciation:
                        word_prons = [
                            process_pronunciation(wp, token) for wp in word_prons
                        ]

                    sentence_prons[word_idx] = [
                        Phonemizer.maybe_separate_tones(wp, separate_tones)
                        for wp in word_prons
                    ]

        if between_words and word_breaks:
            # Add final word break
            sentence_prons.append([[IPA.BREAK_WORD.value]])

        return sentence_prons

    # -------------------------------------------------------------------------

    def predict(
        self, words: typing.Iterable[str], **kwargs
    ) -> typing.Iterable[typing.Tuple[str, typing.List[str]]]:
        """Predict word pronunciations using built-in g2p model"""
        with self.g2p_lock:
            if not self.g2p_model_path.is_file():
                # Look for a gzipped model and extract it
                g2p_gzip_path = Path(str(self.g2p_model_path) + ".gz")
                if g2p_gzip_path.is_file():
                    _LOGGER.debug("Unzipping %s", g2p_gzip_path)
                    with open(self.g2p_model_path, "wb") as out_file:
                        with gzip.open(g2p_gzip_path, "rb") as in_file:
                            shutil.copyfileobj(in_file, out_file)

        # Apply case transformation
        if self.casing:
            words = [self.casing(w) for w in words]

        for result in phonetisaurus.predict(
            words, model_path=self.g2p_model_path, **kwargs
        ):
            yield result

    # -------------------------------------------------------------------------

    # pylint: disable=no-self-use
    def is_word(self, word: str) -> bool:
        """
        Determines whether a word should have its pronunciation guessed.
        Meant to be overridden by the Language class.
        """
        return True

    # -------------------------------------------------------------------------

    def load_lexicon(self):
        """Load lexicon(s) from config"""
        # Load lexicons
        for lexicon_path in self.config.get("lexicons", []):
            if os.path.isfile(lexicon_path):
                _LOGGER.debug("Loading lexicon from %s", lexicon_path)
                with maybe_gzip_open(lexicon_path, "r") as lexicon_file:
                    load_lexicon(lexicon_file, lexicon=self.lexicon, casing=self.casing)
            else:
                _LOGGER.warning("Skipping lexicon at %s", lexicon_path)

        self.lexicon_loaded = True
        _LOGGER.debug("Loaded pronunciations for %s word(s)", len(self.lexicon))

    # -------------------------------------------------------------------------

    @staticmethod
    def maybe_separate_tones(
        word_pron: PRONUNCIATION_TYPE, separate_tones: bool = False
    ) -> PRONUNCIATION_TYPE:
        """If separate_tones is True, tones will be separated out as additional strings"""
        if not separate_tones:
            return word_pron

        new_word_pron = []
        for phoneme_str in word_pron:
            new_phoneme_str = ""
            tone = ""

            in_tone = False
            codepoints = unicodedata.normalize("NFD", phoneme_str)
            for c in codepoints:
                if in_tone and (c in {IPA.TONE_GLOTTALIZED, IPA.TONE_SHORT}):
                    # Interpret as part of tone
                    tone += c
                elif IPA.is_tone(c):
                    tone += c
                    in_tone = True
                else:
                    # Add to phoneme
                    new_phoneme_str += c

            new_word_pron.append(unicodedata.normalize("NFC", new_phoneme_str))
            if tone:
                new_word_pron.append(unicodedata.normalize("NFC", tone))

        return new_word_pron

    @staticmethod
    def remove_nonword_chars(word: str) -> str:
        """Remove non-word characters from a string"""
        return NON_WORD_CHARS.sub("", word)
