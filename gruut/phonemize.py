"""Class for getting phonetic pronunciations for tokenized text"""
import abc
import collections
import re
import sqlite3
import typing

from gruut_ipa import IPA

from .const import REGEX_TYPE, TOKEN_OR_STR, WORD_PHONEMES, Token, WordPronunciation
from .utils import maybe_compile_regex

# -----------------------------------------------------------------------------


class Phonemizer(abc.ABC):
    """Base class for phonemizers"""

    def pre_phonemize(self, tokens: typing.Sequence[Token]) -> typing.Sequence[Token]:
        """Pre-process tokens before phonemization (called in phonemize)"""
        return tokens

    @abc.abstractmethod
    def phonemize(
        self, tokens: typing.Sequence[TOKEN_OR_STR]
    ) -> typing.Iterable[WORD_PHONEMES]:
        """Generate phonetic pronunciation for each token"""
        pass

    def post_phonemize(
        self, token: Token, token_pron: WordPronunciation
    ) -> typing.Sequence[WORD_PHONEMES]:
        """Post-process tokens/pronunciton after phonemization (called in phonemize)"""
        return token_pron.phonemes

    @abc.abstractmethod
    def get_pronunciation(token: TOKEN_OR_STR) -> typing.Optional[WORD_PHONEMES]:
        pass


# -----------------------------------------------------------------------------


class SqlitePhonemizer(Phonemizer):
    """
    Full-featured phonemizer with sqlite database lexicon.

    Pipeline is (roughly):
    1. Cache is checked first for exact word, then filtered word (non-word chars stripped)
    2. Database is checked first for exact word, then filtered word
    3. Pronunciations are guessed with self.guess_pronunciations
    4. The best pronunciation is:
       a. The Nth pronunciation if a word index is provided (1-based)
       b. The pronunciation with the highest count of matching features (e.g., part of speech)
       c. The first pronunciation

    Attributes
    ----------
    database: Union[str, sqlite3.Connection]
        sqlite3 database connection or path to database
        See also: SqlitePhonemizer.create_tables()

    token_features: Optional[Sequence[str]]
        List of feature names that tokens should have.
        Automatically populated from an existing database.
        See also: TokenFeatures
        default: None

    use_word_indexes: bool
        Allow words of the form word_N where N is the 1-based pronunciation index.
        defaut: False

    word_index_pattern: REGEX_TYPE
        Regex used to group words into text (first group) and pronunciation index (second group).
        See also: use_word_indexes
        default: WORD_INDEX_PATTERN

    word_break: Optional[str]
        Phoneme string to insert between words (e.g., "#" for IPA).
        default: None

    minor_breaks: Optional[Union[Collection[str], Mapping[str, str]]]
        List of minor break symbols, or mapping from symbol to phonemes (e.g. {",": "|"}).
        Minor breaks are short pauses in a sentence.
        default: None

    major_breaks: Optional[Union[Collection[str], Mapping[str, str]]]
        List of major break symbols, or mapping from symbol to phonemes (e.g. {".": "‖"}).
        Major breaks are between sentences.
        default: None

    lookup_with_only_words_chars: bool
        If word cannot be found in lexicon, strip non-word characters from it and try again.
        See also: non_word_chars_pattern
        default: False

    non_word_chars_pattern: REGEX_TYPE
        Pattern used to strip non-word characters from a word.
        See also: lookup_with_only_words_chars
        default: NON_WORD_CHARS_PATTERN,

    use_features: bool
        If True, multiple pronunciations are disambiguated using token and word pronunciation features.
        The best pronunciation has the most preferred features matching the token.
        default: True

    fail_on_unknown_words: bool
        If True, phonemization will raise an exception if a word does not have a pronunciation.
        default: False

    cache_pronunciations: bool
        Pronunciations are cached in-memory when looked up in the database or guessed.
        default: True

    lexicon: Optional[Mapping[str, Sequence[WordPronunciation]]]
        Pre-existing lexicon used to augment database.
        default: None


    """

    # word_N where N is 1-based pronunciation index
    WORD_INDEX_PATTERN = re.compile(r"^(.+)_(\d+)$")

    # Pattern to match non-word characters
    NON_WORD_CHARS_PATTERN = re.compile(r"\W")

    def __init__(
        self,
        database: typing.Union[str, sqlite3.Connection],
        token_features: typing.Optional[typing.Sequence[str]] = None,
        use_word_indexes: bool = False,
        word_index_pattern: REGEX_TYPE = WORD_INDEX_PATTERN,
        word_break: typing.Optional[str] = None,
        minor_breaks: typing.Optional[
            typing.Union[typing.Collection[str], typing.Mapping[str, str]]
        ] = None,
        major_breaks: typing.Optional[
            typing.Union[typing.Collection[str], typing.Mapping[str, str]]
        ] = None,
        lookup_with_only_words_chars: bool = False,
        non_word_chars_pattern: REGEX_TYPE = NON_WORD_CHARS_PATTERN,
        use_features: bool = True,
        fail_on_unknown_words: bool = False,
        cache_pronunciations: bool = True,
        lexicon: typing.Optional[
            typing.Mapping[str, typing.Sequence[WordPronunciation]]
        ] = None,
    ):
        self.db_conn: typing.Optional[sqlite3.Connection] = database if isinstance(
            database, sqlite3.Connection
        ) else None
        self.db_path: typing.Optional[str] = database if isinstance(
            database, str
        ) else None

        assert self.db_conn or self.db_path, "Database connection or path required"

        self.feature_to_id: typing.Dict[str, int] = {}
        self.id_to_feature: typing.Dict[int, str] = {}

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
        self.minor_breaks: typing.Dict[str, str] = {}
        if minor_breaks:
            if not isinstance(minor_breaks, collections.abc.Mapping):
                for break_str in minor_breaks:
                    self.minor_breaks[break_str] = IPA.BREAK_MINOR
            else:
                self.minor_breaks = minor_breaks

        # break symbol -> break IPA
        self.major_breaks: typing.Dict[str, str] = {}
        if major_breaks:
            if not isinstance(major_breaks, collections.abc.Mapping):
                for break_str in major_breaks:
                    self.major_breaks[break_str] = IPA.BREAK_MAJOR
            else:
                self.major_breaks = major_breaks

        self.cache_pronunciations = cache_pronunciations
        self.lexicon: typing.Mapping[
            str, typing.Sequence[WordPronunciation]
        ] = lexicon if lexicon is not None else {}

    def phonemize(
        self, tokens: typing.Sequence[TOKEN_OR_STR]
    ) -> typing.Iterable[WORD_PHONEMES]:
        """Generate phonetic pronunciation for each token"""
        # Convert strings to tokens
        tokens = [Token(t) if isinstance(t, str) else t for t in tokens]

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
        word_prons = self.lexicon.get(word)

        if self.lookup_with_only_words_chars:
            # Try the cache with only word chars
            word_prons = self.lexicon.get(filtered_word)

        if not word_prons:
            # Try to look up in the database next
            word_prons = list(self.select_prons(word))

            if not word_prons:
                # Try the database with only word chars
                word_prons = list(self.select_prons(filtered_word))

            if word_prons and self.cache_pronunciations:
                # Add loaded pronunciations to the lexicon
                self.lexicon[word] = word_prons

        if not word_prons:
            # Try to guess pronunciation last
            word_prons = list(self.guess_pronunciations(token))

            if word_prons and self.cache_pronunciations:
                # Add guessed pronunciations to the lexicon
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
        """Chooses the best pronunciation for a token from a list"""
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
        """Guess pronunciations for a word missing from the lexicon"""
        return []

    # -------------------------------------------------------------------------

    def _connect(self):
        """Ensure connection to database"""
        if self.db_conn is None:
            assert self.db_path is not None
            self.db_conn = sqlite3.connect(self.db_path)

            if not self.feature_to_id:
                # Try to load feature names from the database
                try:
                    cursor = self.db_conn.execute(
                        "SELECT feature_id, feature FROM feature_names"
                    )

                    for row in cursor:
                        feature_id, feature_name = row[0], row[1]
                        self.feature_to_id[feature_name] = feature_id
                        self.id_to_feature[feature_id] = feature_name

                except Exception:
                    pass

    def create_tables(self, drop_existing: bool = False, commit: bool = True):
        """Create required database tables"""
        self._connect()

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

        cursor = self.db_conn.cursor()

        for pron_idx, word_pron in enumerate(word_prons):
            phonemes = " ".join(word_pron.phonemes)

            cursor.execute(
                "INSERT INTO word_phonemes (word, pron_order, phonemes) VALUES (?, ?, ?)",
                (word, pron_idx, phonemes),
            )

            # Insert preferred features
            pron_id = cursor.lastrowid
            for (feature_name, feature_values) in word_pron.preferred_features.items():
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
        self, word: str, include_features: bool = True
    ) -> typing.Iterable[WordPronunciation]:
        """Look up pronunciations for a word in the database"""
        self._connect()

        cursor = self.db_conn.execute(
            "SELECT id, phonemes FROM word_phonemes WHERE word = ? ORDER by pron_order",
            (word,),
        )
        for row in cursor:
            # Assume phonemes are whitespace-separated
            pron_id, phonemes = row[0], row[1].split()

            preferred_features: typing.Dict[str, typing.Set[str]] = {}
            if include_features and self.id_to_feature:
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

            yield WordPronunciation(
                phonemes=phonemes, preferred_features=preferred_features
            )

    def delete_prons(self, word: str, commit: bool = True):
        """Delete all pronuncations of a word from the database"""
        self._connect()

        self.db_conn.execute("DELETE FROM word_phonemes WHERE word = ?", (word,))

        if commit:
            self.db_conn.commit()


# -----------------------------------------------------------------------------


class UnknownWordError(Exception):
    """Raised when word pronunciation cannot be guessed"""

    def __init__(self, token: TOKEN_OR_STR):
        self.token = token
        self.message = f"Unknown word: {token}"
        super().__init__(self.message)


# -----------------------------------------------------------------------------

# _LOGGER = logging.getLogger("gruut.phonemize")

# WORD_WITH_INDEX = re.compile(r"^([^_]+)_(\d+)$")

# NON_WORD_CHARS = re.compile(r"\W")

# # -----------------------------------------------------------------------------


# class Phonemizer:
#     """Gets phonetic pronunciations for clean words"""

#     def __init__(
#         self,
#         config,
#         lexicon: typing.Optional[LEXICON_TYPE] = None,
#         preload_lexicon: bool = False,
#     ):
#         self.config = config

#         # Short pause symbols (commas, etc.)
#         self.minor_breaks: typing.Set[str] = set(
#             pydash.get(self.config, "symbols.minor_breaks", [])
#         )

#         # End of sentence symbols
#         self.major_breaks: typing.Set[str] = set(
#             pydash.get(self.config, "symbols.major_breaks", [])
#         )

#         # If True, question marks add rising intonation to the previous word
#         self.question_mark = bool(
#             pydash.get(self.config, "symbols.question_mark", False)
#         )

#         self.has_tones = bool(pydash.get(self.config, "language.tones", []))

#         # Case transformation (lower/upper)
#         casing = pydash.get(self.config, "symbols.casing")
#         self.casing: typing.Optional[typing.Callable[[str], str]] = None
#         if casing == "lower":
#             self.casing = str.lower
#         elif casing == "upper":
#             self.casing = str.upper

#         self.g2p_model_path = Path(pydash.get(self.config, "g2p.model"))

#         self.lexicon: LEXICON_TYPE = {}
#         self.lexicon_loaded = False
#         self.lexicon_conn: typing.Optional[sqlite3.Connection] = None

#         if lexicon:
#             # Use supplied object
#             self.lexicon = lexicon
#             self.lexicon_loaded = True
#         else:
#             # Load lexicon from file or database
#             self.load_lexicon(preload=preload_lexicon)

#     # -------------------------------------------------------------------------

#     def phonemize(
#         self,
#         tokens: typing.Sequence[typing.Union[str, Token]],
#         word_indexes: bool = False,
#         guess_word: typing.Optional[
#             typing.Callable[[Token], typing.Optional[typing.List[WordPronunciation]]]
#         ] = None,
#         word_breaks: bool = False,
#         minor_breaks: bool = True,
#         major_breaks: bool = True,
#         guess_with_word_chars: bool = True,
#         use_pos: bool = True,
#         process_pronunciation: typing.Optional[
#             typing.Callable[[WordPronunciation, Token], WordPronunciation]
#         ] = None,
#         fail_on_unknown_words: bool = False,
#     ) -> typing.List[typing.List[WordPronunciation]]:
#         """Get all possible pronunciations for cleaned words"""
#         if not self.lexicon_loaded:
#             # Dynamically load lexicon(s)
#             self.load_lexicon()

#         sentence_prons: typing.List[typing.List[WordPronunciation]] = []
#         missing_words: typing.List[typing.Tuple[int, Token]] = []
#         between_words = False

#         if word_breaks:
#             # Add initial word break
#             sentence_prons.append([WordPronunciation([IPA.BREAK_WORD.value])])

#         for token_or_str in tokens:
#             if isinstance(token_or_str, Token):
#                 token = token_or_str
#             else:
#                 token = Token(text=token_or_str)

#             word = token.text

#             if word in self.minor_breaks:
#                 if word_breaks and between_words:
#                     # Add end of word break
#                     sentence_prons.append([WordPronunciation([IPA.BREAK_WORD.value])])

#                 # Minor break (short pause)
#                 if minor_breaks:
#                     sentence_prons.append([WordPronunciation([IPA.BREAK_MINOR.value])])

#                 between_words = True
#                 continue

#             if word in self.major_breaks:
#                 if word_breaks and between_words:
#                     # Add end of word break
#                     sentence_prons.append([WordPronunciation([IPA.BREAK_WORD.value])])

#                 # Major break (sentence boundary)
#                 if major_breaks:
#                     sentence_prons.append([WordPronunciation([IPA.BREAK_MAJOR.value])])

#                 between_words = False
#                 continue

#             if word_breaks and between_words:
#                 # Add IPA word break symbol between words
#                 sentence_prons.append([WordPronunciation([IPA.BREAK_WORD.value])])

#             # Actual word
#             between_words = True
#             index: typing.Optional[int] = None
#             if word_indexes:
#                 index_match = WORD_WITH_INDEX.match(word)
#                 if index_match:
#                     word = index_match.group(1)
#                     index = int(index_match.group(2))

#             word_prons = self.lookup_word(word)
#             word_guessed = False

#             if not word_prons and guess_with_word_chars:
#                 # Try again with non-word characters removed
#                 filtered_word = Phonemizer.remove_nonword_chars(word)
#                 word_prons = self.lookup_word(filtered_word)

#             if not word_prons and guess_word:
#                 # Use supplied function
#                 word_guessed = True
#                 word_prons = guess_word(token)
#                 if word_prons is not None:
#                     # Update lexicon
#                     self.lexicon[word] = word_prons

#             if word_prons:
#                 # Language-specific processing
#                 if process_pronunciation:
#                     word_prons = [process_pronunciation(wp, token) for wp in word_prons]

#                 # In lexicon
#                 if index is None:
#                     if use_pos and token.pos:
#                         word_prons = sorted(
#                             word_prons,
#                             key=lambda wp: 1
#                             # pylint: disable=cell-var-from-loop
#                             if (wp.valid_pos and (token.pos in wp.valid_pos)) else 0,
#                             reverse=True,
#                         )

#                     # All pronunciations
#                     sentence_prons.append(word_prons)
#                 else:
#                     # Specific pronunciation.
#                     # Clamp 1-based index.
#                     pron_index = max(0, index - 1) % len(word_prons)
#                     sentence_prons.append([word_prons[pron_index]])
#             else:
#                 if not word_guessed and self.is_word(token.text):
#                     # Need to guess
#                     missing_words.append((len(sentence_prons), token))

#                 # Add placeholder
#                 sentence_prons.append([])

#         words_to_guess = set(t.text for _, t in missing_words)

#         if words_to_guess:
#             _LOGGER.debug("Guessing pronunciations for %s", words_to_guess)
#             guessed_words: typing.Set[str] = set()

#             for word, word_phonemes in self.predict(words=words_to_guess):
#                 # Add to lexicon
#                 self.lexicon[word] = [WordPronunciation(word_phonemes)]
#                 guessed_words.add(word)

#             if fail_on_unknown_words:
#                 # Verify that all missing words were guessed
#                 failed_words = words_to_guess - guessed_words
#                 if failed_words:
#                     raise UnknownWordsError(failed_words)

#             # Fill in missing words
#             for word_idx, token in missing_words:
#                 word_prons = self.lookup_word(token.text)
#                 if word_prons:
#                     # Language-specific processing
#                     if process_pronunciation:
#                         word_prons = [
#                             process_pronunciation(wp, token) for wp in word_prons
#                         ]

#                     sentence_prons[word_idx] = word_prons

#         if between_words and word_breaks:
#             # Add final word break
#             sentence_prons.append([WordPronunciation([IPA.BREAK_WORD.value])])

#         return sentence_prons

#     # -------------------------------------------------------------------------

#     def predict(
#         self, words: typing.Iterable[str], **kwargs
#     ) -> typing.Iterable[typing.Tuple[str, typing.List[str]]]:
#         """Predict word pronunciations using built-in g2p model"""
#         # Apply case transformation
#         if self.casing:
#             words = [self.casing(w) for w in words]

#         for result in phonetisaurus.predict(
#             words, model_path=self.g2p_model_path, **kwargs
#         ):
#             yield result

#     # -------------------------------------------------------------------------

#     # pylint: disable=no-self-use
#     def is_word(self, word: str) -> bool:
#         """
#         Determines whether a word should have its pronunciation guessed.
#         Meant to be overridden by the Language class.
#         """
#         return True

#     # -------------------------------------------------------------------------

#     def load_lexicon(self, preload: bool = False):
#         """Load lexicon from config"""
#         maybe_lexicon_path = self.config.get("lexicon")
#         if maybe_lexicon_path:
#             lexicon_path = Path(maybe_lexicon_path)
#             _LOGGER.debug("Loading lexicon from %s", lexicon_path)

#             if lexicon_path.is_file():
#                 if lexicon_path.suffix == ".db":
#                     # Sqlite3 database
#                     self.lexicon_conn = sqlite3.connect(lexicon_path)
#                     if preload:
#                         # Load entire database
#                         _LOGGER.debug("Pre-loading lexicon from database")
#                         num_pronunciations = 0
#                         cursor = self.lexicon_conn.execute(
#                             "SELECT word, phonemes, pos FROM word_phonemes ORDER by pron_order"
#                         )
#                         for row in cursor:
#                             word, phonemes, pos_list = (
#                                 row[0],
#                                 row[1].split(),
#                                 row[2].split(),
#                             )
#                             word_prons = self.lexicon.get(word)
#                             if not word_prons:
#                                 word_prons = []
#                                 self.lexicon[word] = word_prons

#                             valid_pos: typing.Optional[typing.Set[str]] = None
#                             if pos_list:
#                                 valid_pos = set(pos_list)

#                             word_prons.append(
#                                 WordPronunciation(
#                                     phonemes=phonemes, valid_pos=valid_pos
#                                 )
#                             )
#                             num_pronunciations += 1

#                         _LOGGER.debug(
#                             "Loaded %s pronuncation(s) for %s word(s)",
#                             num_pronunciations,
#                             len(self.lexicon),
#                         )
#                         self.lexicon_loaded = True
#                 else:
#                     # Text file
#                     with maybe_gzip_open(lexicon_path, "r") as lexicon_file:
#                         load_lexicon(
#                             lexicon_file, lexicon=self.lexicon, casing=self.casing
#                         )

#                     _LOGGER.debug(
#                         "Loaded pronunciations for %s word(s)", len(self.lexicon)
#                     )

#         self.lexicon_loaded = True

#     # -------------------------------------------------------------------------

#     def lookup_word(self, word: str) -> typing.Optional[typing.List[WordPronunciation]]:
#         """Look up word in lexicon first, database second"""
#         word_prons = self.lexicon.get(word)
#         if not word_prons and self.lexicon_conn:
#             # Look up in database
#             cursor = self.lexicon_conn.execute(
#                 "SELECT phonemes, pos FROM word_phonemes WHERE word = ? ORDER by pron_order",
#                 (word,),
#             )
#             for row in cursor:
#                 phonemes = row[0].split()
#                 if phonemes:
#                     if not word_prons:
#                         word_prons = []

#                     pos_list = row[1].split()
#                     valid_pos: typing.Optional[typing.Set[str]] = None
#                     if pos_list:
#                         valid_pos = set(pos_list)

#                     word_prons.append(
#                         WordPronunciation(phonemes=phonemes, valid_pos=valid_pos)
#                     )

#             if word_prons:
#                 # Update lexicon
#                 self.lexicon[word] = word_prons

#         return word_prons

#     # -------------------------------------------------------------------------

#     def post_process_sentence(
#         self,
#         language: str,
#         tokens: typing.List[Token],
#         token_phonemes: typing.List[PHONEMES_TYPE],
#         word_breaks: bool = False,
#     ) -> typing.List[PHONEMES_TYPE]:
#         """Do language-specific post-processing on an entire sentence"""
#         if language == "fr-fr":
#             return list(fr_liason(tokens, token_phonemes, word_breaks=word_breaks))

#         # No post-processing
#         return token_phonemes

#     # -------------------------------------------------------------------------

#     @staticmethod
#     def remove_nonword_chars(word: str) -> str:
#         """Remove non-word characters from a string"""
#         return NON_WORD_CHARS.sub("", word)


# # -----------------------------------------------------------------------------
