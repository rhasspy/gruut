"""Class for getting phonetic pronunciations for tokenized text"""
import abc
import collections
import logging
import re
import sqlite3
import threading
import typing
from pathlib import Path

from gruut_ipa import IPA

from .const import REGEX_TYPE, TOKEN_OR_STR, WORD_PHONEMES, Token, WordPronunciation
from .g2p import GraphemesToPhonemes
from .utils import maybe_compile_regex

_LOGGER = logging.getLogger("gruut.phonemize")

# -----------------------------------------------------------------------------


class Phonemizer(abc.ABC):
    """Abstract base class for phonemizers"""

    # pylint: disable=R0201
    def pre_phonemize(self, tokens: typing.Sequence[Token]) -> typing.Sequence[Token]:
        """
        Pre-process tokens before phonemization (called in :py:meth:`phonemize`).

        Args:
            tokens: Tokens to pre-process

        Returns:
            Pre-processed tokens
        """
        return tokens

    @abc.abstractmethod
    def phonemize(
        self, tokens: typing.Sequence[TOKEN_OR_STR]
    ) -> typing.Iterable[WORD_PHONEMES]:
        """
        Generate phonetic pronunciation for each token.

        Args:
            tokens: Tokens to generate pronunciations for

        Returns:
            List of phonemes for each token
        """
        pass

    # pylint: disable=R0201
    def post_phonemize(
        self, token: Token, token_pron: WordPronunciation
    ) -> WORD_PHONEMES:
        """
        Post-process tokens/pronunciton after phonemization (called in phonemize).

        Args:
            token: Token to post-process
            token_pron: Phonetic pronunciation of word

        Returns:
            Post-processed list of phonemes
        """
        return token_pron.phonemes

    @abc.abstractmethod
    def get_pronunciation(
        self, token: TOKEN_OR_STR
    ) -> typing.Optional[WordPronunciation]:
        """
        Gets the best pronunciation for a token (or None).

        Args:
            token: Token to get phonetic pronunciation for

        Returns:
            Phonetic pronunciation or None (if not available or cannot be guessed)
        """
        pass


# -----------------------------------------------------------------------------


class SqlitePhonemizer(Phonemizer):
    """
    Full-featured phonemizer with sqlite database lexicon backend.

    Pipeline is (roughly):

    #. Cache is checked first for exact word, then filtered word (non-word chars stripped)
    #. Database is checked first for exact word, then filtered word
    #. Pronunciations are guessed with self.guess_pronunciations
    #. The best pronunciation is:

       * The Nth pronunciation if a word index is provided (1-based)
       * The pronunciation with the highest count of matching features (e.g., part of speech)
       * The first pronunciation
       * Guessed using pre-trained gruut.g2p model

    Attributes:
        database_path: Path to sqlite3 database
        token_features: List of feature names that tokens should have. Automatically populated from an existing database.
        use_word_indexes: Allow words of the form "word_N" where N is the 1-based pronunciation index
        word_index_pattern: Regex used to group words into text (first group) and pronunciation index (second group).
        word_break: Phoneme string to insert between words (e.g., "#" for IPA)
        ipa_minor_breaks: If `True`, replace minor breaks with `gruut_ipa.IPA.BREAK_MINOR` symbol. If `False`, the original symbol is used (e.g., ",")
        minor_breaks: List of minor break symbols, or mapping from symbol to phonemes (e.g. {",": "|"}). Minor breaks are short pauses in a sentence.
        ipa_major_breaks: If `True`, replace major breaks with `gruut_ipa.IPA.BREAK_MAJOR` symbol. If `False`, the original symbol is used (e.g., ".")
        major_breaks: List of major break symbols, or mapping from symbol to phonemes (e.g. {".": "‖"}). Major breaks are between sentences.
        lookup_with_only_words_chars: If word cannot be found in lexicon, strip non-word characters from it and try again
        non_word_chars_pattern: Regex pattern used to strip non-word characters from a word
        use_features: If `True`, multiple pronunciations are disambiguated using token and word pronunciation features. The best pronunciation has the most preferred features matching the token.
        fail_on_unknown_words: If True, phonemization will raise an :py:class:`UnknownWordError` if a word does not have a pronunciation
        cache_pronunciations: Pronunciations are cached in-memory when looked up in the database or guessed
        lexicon: Pre-existing lexicon used to augment database (modified by phonemizer)
        g2p_model: Path to pre-trained grapheme-to-phoneme model. See also: :py:mod:`gruut.g2p`
        feature_map: Mapping from feature name (e.g. "pos") to a mapping from raw feature values to normalized values (e.g., "NNS": "NN"). Used to simplify the feature values needed in the lexicon to disambiguate pronunciations.
        remove_stress: If `True`, remove stress characters (ˈˌ) when loading/guessing prounciations
        remove_accents: If `True`, remove accent characters ('²) when loading/guessing prounciations. If None, use value of return_stress.
    """

    WORD_INDEX_PATTERN = re.compile(r"^(.+)_(\d+)$")
    """word_N where N is 1-based pronunciation index"""

    NON_WORD_CHARS_PATTERN = re.compile(r"\W")
    """Pattern to match non-word characters"""

    def __init__(
        self,
        database_path: typing.Union[str, Path],
        token_features: typing.Optional[typing.Sequence[str]] = None,
        use_word_indexes: bool = False,
        word_index_pattern: REGEX_TYPE = WORD_INDEX_PATTERN,
        word_break: typing.Optional[str] = None,
        ipa_minor_breaks: bool = True,
        minor_breaks: typing.Optional[
            typing.Union[typing.Collection[str], typing.Mapping[str, str]]
        ] = None,
        ipa_major_breaks: bool = True,
        major_breaks: typing.Optional[
            typing.Union[typing.Collection[str], typing.Mapping[str, str]]
        ] = None,
        lookup_with_only_words_chars: bool = False,
        non_word_chars_pattern: REGEX_TYPE = NON_WORD_CHARS_PATTERN,
        use_features: bool = True,
        fail_on_unknown_words: bool = False,
        cache_pronunciations: bool = True,
        lexicon: typing.Optional[
            typing.MutableMapping[str, typing.List[WordPronunciation]]
        ] = None,
        g2p_model: typing.Optional[typing.Union[str, Path]] = None,
        feature_map: typing.Optional[
            typing.Mapping[str, typing.Mapping[str, str]]
        ] = None,
        remove_stress: bool = False,
        remove_accents: typing.Optional[bool] = None,
    ):
        # Thread-local variables
        self.thread_local = threading.local()

        self.db_path = str(database_path)

        self.feature_to_id: typing.Dict[str, int] = {}
        self.id_to_feature: typing.Dict[int, str] = {}
        self.load_save_features: bool = True

        if token_features:
            # Generate feature id <-> name maps
            for feature_index, feature_name in enumerate(token_features):
                self.feature_to_id[feature_name] = feature_index
                self.id_to_feature[feature_index] = feature_name

        self.use_word_indexes = use_word_indexes
        self.word_break = word_break
        self.lookup_with_only_words_chars = lookup_with_only_words_chars
        self.word_index_pattern = maybe_compile_regex(word_index_pattern)
        self.non_word_chars_pattern = maybe_compile_regex(non_word_chars_pattern)
        self.use_features = use_features
        self.fail_on_unknown_words = fail_on_unknown_words

        # break symbol -> break IPA
        self.minor_breaks: typing.Mapping[str, str] = {}
        if minor_breaks:
            if not isinstance(minor_breaks, collections.abc.Mapping):
                for break_str in minor_breaks:
                    self.minor_breaks[break_str] = (
                        IPA.BREAK_MINOR.value if ipa_minor_breaks else break_str
                    )
            else:
                self.minor_breaks = minor_breaks

        # break symbol -> break IPA
        self.major_breaks: typing.Mapping[str, str] = {}
        if major_breaks:
            if not isinstance(major_breaks, collections.abc.Mapping):
                for break_str in major_breaks:
                    self.major_breaks[break_str] = (
                        IPA.BREAK_MAJOR.value if ipa_major_breaks else break_str
                    )
            else:
                self.major_breaks = major_breaks

        self.cache_pronunciations = cache_pronunciations
        self.lexicon: typing.MutableMapping[
            str, typing.List[WordPronunciation]
        ] = lexicon if lexicon is not None else {}

        self.g2p_tagger: typing.Optional[GraphemesToPhonemes] = None
        if g2p_model is not None:
            # Load g2p model
            self.g2p_tagger = GraphemesToPhonemes(g2p_model)

        self.feature_map = feature_map

        # Create str.translate table for phoneme cleaning
        self.phoneme_translate: typing.Optional[
            typing.Mapping[typing.Any, typing.Any]
        ] = None

        drop_phonemes = []
        remove_accents = remove_stress if remove_accents is None else remove_accents

        if remove_stress:
            # Primary/secondary stress (ˈˌ)
            drop_phonemes.extend(["\u02C8", "\u02CC"])

        if remove_accents:
            # Accute/grave accents ('²)
            drop_phonemes.extend(["\u0027", "\u00B2"])

        if drop_phonemes:
            # Third argument maps characters to None
            self.phoneme_translate = str.maketrans("", "", "".join(drop_phonemes))

        # Lock used for lexicon modification
        self.lexicon_lock = threading.RLock()

        # Lock used when writing to database
        self.db_write_lock = threading.Lock()

        # True if pronunciations have all been loaded from the database
        self.prons_preloaded = False

    def phonemize(
        self, tokens: typing.Sequence[TOKEN_OR_STR]
    ) -> typing.Iterable[WORD_PHONEMES]:
        # Convert strings to tokens
        tokens = typing.cast(
            typing.List[Token], [Token(t) if isinstance(t, str) else t for t in tokens]
        )

        if self.word_break:
            yield [self.word_break]

        for token in self.pre_phonemize(tokens):
            if token.text in self.minor_breaks:
                # Minor break (short pause)
                yield [self.minor_breaks[token.text]]

                if self.word_break:
                    yield [self.word_break]
            elif token.text in self.major_breaks:
                # Major break (end of sentence)
                yield [self.major_breaks[token.text]]
            else:
                # Word
                token_pron = self.get_pronunciation(token)
                if token_pron:
                    # Has a pronunciation
                    yield self.post_phonemize(token, token_pron)

                    if self.word_break:
                        yield [self.word_break]
                else:
                    # Missing pronunciation
                    if self.fail_on_unknown_words:
                        # Unknown word
                        raise UnknownWordError(token)

                    # No pronunciation
                    yield []

                    if self.word_break:
                        yield [self.word_break]

    def get_pronunciation(
        self, token: TOKEN_OR_STR
    ) -> typing.Optional[WordPronunciation]:
        """Get a single pronunciation for a token or None"""
        if isinstance(token, str):
            # Wrap text
            token = Token(token)

        word = token.text

        # Word with all "non-word" characters removed.
        # Skip if lookup_with_only_words_chars is False.
        filtered_word = (
            self.non_word_chars_pattern.sub("", word)
            if self.lookup_with_only_words_chars
            else word
        )

        word_index: typing.Optional[int] = None

        if self.use_word_indexes:
            # Check if word has explicit index.
            # word_2 means use the second pronunciation (index 1).
            word_match = self.word_index_pattern.match(word)
            if word_match:
                word = word_match.group(1)
                word_index = max(0, int(word_match.group(2)) - 1)

        # Try to look up in the lexicon cache first
        with self.lexicon_lock:
            word_prons = self.lexicon.get(word)

            if self.lookup_with_only_words_chars:
                # Try the cache with only word chars
                word_prons = self.lexicon.get(filtered_word)

        if not word_prons:
            # Try to look up in the database next
            word_prons = list(pron for _word, pron in self.select_prons(word))

            if not word_prons:
                # Try the database with only word chars
                word_prons = list(
                    pron for _word, pron in self.select_prons(filtered_word)
                )

            if word_prons and self.cache_pronunciations:
                # Add loaded pronunciations to the lexicon
                with self.lexicon_lock:
                    self.lexicon[word] = word_prons

        if not word_prons:
            # Try to guess pronunciation last
            word_prons = list(self.guess_pronunciations(token))

            if word_prons and self.cache_pronunciations:
                # Add guessed pronunciations to the lexicon
                with self.lexicon_lock:
                    self.lexicon[word] = word_prons

        if word_prons:
            # Determine best pronunciation from the list
            return self.get_best_pronunciation(token, word_prons, word_index=word_index)

        # No pronunciation
        return None

    def get_best_pronunciation(
        self,
        token: Token,
        word_prons: typing.Sequence[WordPronunciation],
        word_index: typing.Optional[int] = None,
    ) -> typing.Optional[WordPronunciation]:
        """
        Chooses the best pronunciation for a token from a list.

        The method is fairly simple:

        #. If a word_index is supplied, that specific pronunciation is chosen
        #. Otherwise, the pronunciation with the most matching features is chosen

           * For example, if the token has a "NOUN" part of speech tag in its :py:attr:`~gruut.const.Token.features` and one word pronunciation has a matching "NOUN" value in its :py:attr:`~gruut.const.WordPronunciation.preferred_features`, then it will be chosen

        #. Otherwise, the first pronunciation is chosen

        Args:
            token: The token to choose a pronunciation for
            word_prons: Available pronunciations
            word_index: Optional 1-based index of desired pronunciation

        Returns:
            Best pronunciation or None if one is not available
        """
        if word_prons:
            if word_index is not None:
                # Use explicit index
                real_index = max(0, min(len(word_prons) - 1, word_index))
                return word_prons[real_index]

            if self.use_features and token.features:
                # Find pronunciation with all matching features
                best_pron: typing.Optional[WordPronunciation] = None
                best_num_preferred = 0

                for word_pron in word_prons:
                    num_preferred = 0

                    # Count number of matching feature values
                    for feature_name, feature_value in token.features.items():
                        if self.feature_map:
                            # Look up normalize value.
                            # For example: anything noun-like becomes just NN.
                            normalized_map = self.feature_map.get(feature_name)

                            if normalized_map:
                                feature_value = normalized_map.get(
                                    feature_value, feature_value
                                )

                        preferred_values = word_pron.preferred_features.get(
                            feature_name
                        )
                        if preferred_values and (feature_value in preferred_values):
                            num_preferred += 1

                    if num_preferred > best_num_preferred:
                        # Best pronunciation has most matching features
                        best_pron = word_pron
                        best_num_preferred = num_preferred

                if best_pron is not None:
                    return best_pron

            # First pronunciation
            return word_prons[0]

        # No pronunciaton
        return None

    def guess_pronunciations(self, token: Token) -> typing.Iterable[WordPronunciation]:
        """
        Guess pronunciations for a word missing from the lexicon using a :py:class:`gruut.g2p` model.

        Args:
            token: Token whose pronunciation should be guessed

        Returns:
            Zero or more guessed pronunciations

        """
        if self.g2p_tagger:
            _LOGGER.debug("Guessing pronunciations for %s", token)
            guessed_phonemes = self.g2p_tagger(token.text)
            if guessed_phonemes:
                guessed_phonemes = self.clean_phonemes(guessed_phonemes)

                # Single pronunciation for now
                return [WordPronunciation(guessed_phonemes)]

        return []

    def clean_phonemes(self, phonemes: typing.Sequence[str]) -> typing.Sequence[str]:
        """Filter out phonemes using phoneme translation table"""
        if self.phoneme_translate:
            # Clean phonemes and filter out empty ones
            phonemes = [p.translate(self.phoneme_translate) for p in phonemes]
            return [p for p in phonemes if p]

        return phonemes

    # -------------------------------------------------------------------------

    @property
    def db_conn(self) -> typing.Optional[sqlite3.Connection]:
        """Get thread-local database connection"""
        return typing.cast(
            typing.Optional[sqlite3.Connection],
            getattr(self.thread_local, "db_conn", None),
        )

    @db_conn.setter
    def db_conn(self, value: sqlite3.Connection):
        """Set thread-local database connection"""
        self.thread_local.db_conn = value

    def _connect(self):
        """Ensure connection to database"""
        if self.db_conn is None:
            assert self.db_path is not None, "No sqlite3 database path"

            _LOGGER.debug("Connecting to %s", self.db_path)
            self.db_conn = sqlite3.connect(self.db_path)

            if not self.feature_to_id:
                # Try to load feature names from the database
                try:
                    _LOGGER.debug("Attempting to load feature names")
                    cursor = self.db_conn.execute(
                        "SELECT feature_id, feature FROM feature_names"
                    )

                    for row in cursor:
                        feature_id, feature_name = row[0], row[1]
                        self.feature_to_id[feature_name] = feature_id
                        self.id_to_feature[feature_id] = feature_name

                except Exception:
                    _LOGGER.debug(
                        "Failed to load feature names. Disabling feature loading/saving."
                    )
                    self.load_save_features = False

    def create_tables(self, drop_existing: bool = False, commit: bool = True):
        """Create required database tables"""
        self._connect()

        assert self.db_conn is not None

        with self.db_write_lock:
            if drop_existing:
                self.db_conn.execute("DROP TABLE IF EXISTS word_phonemes")
                self.db_conn.execute("DROP TABLE IF EXISTS feature_names")
                self.db_conn.execute("DROP TABLE IF EXISTS pron_features")

            # Word/phoneme pairs with explicit ordering
            self.db_conn.execute(
                "CREATE TABLE word_phonemes "
                + "(id INTEGER PRIMARY KEY AUTOINCREMENT, word TEXT, pron_order INTEGER, phonemes TEXT);"
            )

            # Names of all features (token_features)
            self.db_conn.execute(
                "CREATE TABLE feature_names "
                + "(id INTEGER PRIMARY KEY AUTOINCREMENT, feature_id INTEGER, feature TEXT);"
            )

            # Insert known features
            for feature_id, feature_name in self.id_to_feature.items():
                self.db_conn.execute(
                    "INSERT INTO feature_names (feature_id, feature) VALUES (?, ?)",
                    (feature_id, feature_name),
                )

            # Feature values for each pronunciation
            self.db_conn.execute(
                "CREATE TABLE pron_features "
                + "(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + "pron_id INTEGER, "
                + "feature_id INTEGER, "
                + "feature_value TEXT, "
                + "FOREIGN KEY(pron_id) REFERENCES word_phonemes(id), "
                + "FOREIGN KEY(feature_id) REFERENCES feature_names(feature_id) "
                + ");"
            )

            if commit:
                self.db_conn.commit()

    def insert_prons(
        self,
        word: str,
        word_prons: typing.Iterable[WordPronunciation],
        commit: bool = True,
    ):
        """Insert pronunciations for a word into the database"""
        self._connect()

        assert self.db_conn is not None

        with self.db_write_lock:
            cursor = self.db_conn.cursor()

            for pron_idx, word_pron in enumerate(word_prons):
                phonemes = " ".join(word_pron.phonemes)

                cursor.execute(
                    "INSERT INTO word_phonemes (word, pron_order, phonemes) VALUES (?, ?, ?)",
                    (word, pron_idx, phonemes),
                )

                if self.load_save_features:
                    # Insert preferred features
                    pron_id = cursor.lastrowid
                    for (
                        feature_name,
                        feature_values,
                    ) in word_pron.preferred_features.items():
                        feature_id = self.feature_to_id.get(feature_name)
                        assert (
                            feature_id is not None
                        ), f"Unknown feature {feature_name} in {self.feature_to_id.keys()}"

                        for feature_value in feature_values:
                            cursor.execute(
                                "INSERT INTO pron_features (pron_id, feature_id, feature_value) VALUES (?, ?, ?)",
                                (pron_id, feature_id, feature_value),
                            )

            if commit:
                self.db_conn.commit()

    def select_prons(
        self,
        word: typing.Optional[typing.Union[str, typing.Sequence[str]]] = None,
        include_features: bool = True,
    ) -> typing.Iterable[typing.Tuple[str, WordPronunciation]]:
        """
        Look up pronunciations for one word (str), many words (list), or all words (None) in the database.

        Returns:
            word, pronunciation tuples
        """
        self._connect()

        assert self.db_conn is not None

        if word is None:
            # All words
            cursor = self.db_conn.execute(
                "SELECT id, word, phonemes FROM word_phonemes ORDER by word, pron_order"
            )
        elif isinstance(word, str):
            # One word
            cursor = self.db_conn.execute(
                "SELECT id, word, phonemes FROM word_phonemes WHERE word = ? ORDER by pron_order",
                (word,),
            )
        else:
            # Many words
            question_marks = ",".join("?" * len(word))
            cursor = self.db_conn.execute(
                f"SELECT id, word, phonemes FROM word_phonemes WHERE word IN ({question_marks}) ORDER by pron_order",
                tuple(word),
            )

        for row in cursor:
            # Assume phonemes are whitespace-separated
            pron_id, row_word, phonemes_str = row[0], row[1], row[2]

            phonemes = phonemes_str.split()
            phonemes = self.clean_phonemes(phonemes)

            preferred_features: typing.Dict[str, typing.Set[str]] = {}

            if include_features and self.load_save_features and self.id_to_feature:
                # Load preferred features for pronunciation
                feature_cursor = self.db_conn.execute(
                    "SELECT feature_id, feature_value FROM pron_features WHERE pron_id = ?",
                    (pron_id,),
                )

                # Accumulate features
                for feature_row in feature_cursor:
                    feature_id, feature_value = feature_row[0], feature_row[1]
                    feature_name = self.id_to_feature[feature_id]

                    feature_values = preferred_features.get(feature_name)
                    if feature_values is None:
                        feature_values = set()
                        preferred_features[feature_name] = feature_values

                    feature_values.add(feature_value)

            yield (
                row_word,
                WordPronunciation(
                    phonemes=phonemes, preferred_features=preferred_features
                ),
            )

    def delete_prons(self, word: str, commit: bool = True):
        """Delete all pronuncations of a word from the database"""
        self._connect()

        assert self.db_conn is not None

        with self.db_write_lock:
            self.db_conn.execute("DELETE FROM word_phonemes WHERE word = ?", (word,))

            if commit:
                self.db_conn.commit()

    def preload_prons(
        self, include_features: bool = True, skip_if_already_loaded: bool = True
    ):
        """Pre-load all pronunciations from the database"""
        if self.prons_preloaded and skip_if_already_loaded:
            return

        self._connect()
        assert self.db_conn is not None

        if include_features and self.load_save_features and self.id_to_feature:
            # With features
            sql = (
                "SELECT word_phonemes.id, word_phonemes.word, word_phonemes.phonemes, pron_features.feature_id, pron_features.feature_value "
                + "FROM word_phonemes LEFT JOIN pron_features ON word_phonemes.id = pron_features.pron_id "
                + "ORDER BY word_phonemes.id, word_phonemes.word, word_phonemes.pron_order"
            )
        else:
            # Without features
            sql = (
                "SELECT id, word, phonemes FROM word_phonemes ORDER BY word, pron_order"
            )

        cursor = self.db_conn.execute(sql)

        with self.lexicon_lock:
            last_pron_id = None
            last_pron = None

            for row in cursor:
                pron_id, word, phonemes_str = row[0], row[1], row[2]

                phonemes = phonemes_str.split()
                phonemes = self.clean_phonemes(phonemes)

                if pron_id != last_pron_id:
                    last_pron = WordPronunciation(phonemes)

                    word_prons = self.lexicon.get(word)
                    if word_prons is None:
                        word_prons = []
                        self.lexicon[word] = word_prons

                    word_prons.append(last_pron)
                    last_pron_id = pron_id

                if len(row) > 3:
                    # Add features
                    assert last_pron is not None
                    last_pron.preferred_features = last_pron.preferred_features or {}

                    feature_id, feature_value = row[3], row[4]
                    feature_name = self.id_to_feature[feature_id]

                    feature_values = last_pron.preferred_features.get(feature_name)
                    if feature_values is None:
                        feature_values = set()
                        last_pron.preferred_features[feature_name] = feature_values

                    feature_values.add(feature_value)

        self.prons_preloaded = True


# -----------------------------------------------------------------------------


class UnknownWordError(Exception):
    """Raised when word pronunciation cannot be guessed"""

    def __init__(self, token: TOKEN_OR_STR):
        self.token = token
        self.message = f"Unknown word: {token}"
        super().__init__(self.message)
