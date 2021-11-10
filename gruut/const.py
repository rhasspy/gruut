"""Shared classes, types, and enums"""
import itertools
import operator
import re
import typing
import xml.etree.ElementTree as etree
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

import babel
import babel.core
import babel.numbers

# alias -> full language name
LANG_ALIASES = {
    "ar": "ar",
    "cs": "cs-cz",
    "de": "de-de",
    "en": "en-us",
    "en-gb": "en-us",
    "es": "es-es",
    "es-mx": "es-es",
    "fa": "fa",
    "fr": "fr-fr",
    "it": "it-it",
    "nl": "nl",
    "nl-nl": "nl",
    "pt-br": "pt",
    "ru": "ru-ru",
    "sv": "sv-se",
    "sw": "sw",
    "zh": "zh-cn",
}

ENGLISH_LANGS = {"en-us", "en-gb"}

# Languages that are expected to have a model directory
KNOWN_LANGS = set(itertools.chain(ENGLISH_LANGS, LANG_ALIASES.values()))


try:
    # Python >= 3.7
    REGEX_PATTERN = re.Pattern  # type: ignore
    REGEX_MATCH = re.Match  # type: ignore
    REGEX_TYPE = typing.Union[str, re.Pattern]  # type: ignore
except AttributeError:
    # Python 3.6
    REGEX_PATTERN = typing.Pattern  # type: ignore
    REGEX_MATCH = typing.Match  # type: ignore
    REGEX_TYPE = typing.Union[str, typing.Pattern]  # type: ignore


# Phonemes for a single word
PHONEMES_TYPE = typing.Sequence[str]

# Type of nodes in a text graph
NODE_TYPE = int

# Property used to hold node data in text graph
DATA_PROP = "data"


class GraphType:
    """Type wrapper for networkx graph"""

    nodes: typing.Dict[NODE_TYPE, typing.Dict[typing.Any, typing.Any]]
    """Get node data for the graph"""

    def add_node(self, node: NODE_TYPE, **kwargs):
        """Add a new node to the graph"""
        pass

    def add_edge(self, src: NODE_TYPE, dst: NODE_TYPE):
        """Add a new edge to the graph"""
        pass

    def out_degree(self, node: NODE_TYPE) -> int:
        """Get number of outgoing edges from a node"""
        pass

    def successors(self, node: NODE_TYPE) -> typing.Iterable[NODE_TYPE]:
        """Yield nodes on outgoing edges"""
        pass

    def predecessors(self, node: NODE_TYPE) -> typing.Iterable[NODE_TYPE]:
        """Yield nodes from incoming edges"""
        pass

    def out_edges(
        self, node: NODE_TYPE
    ) -> typing.Iterable[typing.Tuple[NODE_TYPE, NODE_TYPE]]:
        """Yield outgoing edges from a node"""
        pass

    def add_edges_from(
        self, edges: typing.Iterable[typing.Tuple[NODE_TYPE, NODE_TYPE]]
    ):
        """Add edges from iterable"""
        pass

    def remove_edges_from(
        self, edges: typing.Iterable[typing.Tuple[NODE_TYPE, NODE_TYPE]]
    ):
        """Remove edges from iterable"""
        pass

    def __len__(self) -> int:
        """Get number of nodes in the graph"""
        pass


# -----------------------------------------------------------------------------

DEFAULT_SPLIT_PATTERN = re.compile(r"(\s+)")

NORMALIZE_WHITESPACE_PATTERN = re.compile(r"\s+")
SURROUNDING_WHITESPACE_PATTERN = re.compile(r"^(\s*)\S+(\s*)$")
HAS_DIGIT_PATTERN = re.compile(r"[0-9]")


@dataclass
class Time:
    """Parsed time from text"""

    hours: int
    minutes: int = 0

    period: typing.Optional[str] = None
    """A.M. or P.M."""


# -----------------------------------------------------------------------------


class InterpretAs(str, Enum):
    """Supported options for interpret-as attribute of <say-as>"""

    SPELL_OUT = "spell-out"
    """Word should be spelled out (abc = a b c)"""

    DATE = "date"
    """Word should be interpreted as a date"""

    NUMBER = "number"
    """Word should be interpreted as a number"""

    CURRENCY = "currency"
    """Word should be interpreted as an amount of currency"""

    TIME = "time"
    """Word should be interpreted as a time on the clock"""

    WORD = "word"
    """Interpret as regular word"""


class InterpretAsFormat(str, Enum):
    """Supported options for format attribute of <say-as>"""

    NUMBER_CARDINAL = "cardinal"
    """Cardinal version of number (1 = one)"""

    NUMBER_ORDINAL = "ordinal"
    """Ordinal version of number (1 = first)"""

    NUMBER_DIGITS = "digits"
    """Number as digits (12 = one two)"""

    NUMBER_YEAR = "year"
    """Number as a year (2021 = twenty twenty-one)"""

    # Date formats
    # d = day
    # m = month
    # y = year
    # o = ordinal day ("first" instead of "one")
    DATE_DMY = "dmy"
    DATE_MDY = "mdy"
    DATE_YMD = "ymd"
    DATE_DMY_ORDINAL = "omy"
    DATE_MDY_ORDINAL = "moy"
    DATE_YMD_ORDINAL = "ymo"
    DATE_YM = "ym"
    DATE_MY = "my"
    DATE_MD = "md"
    DATE_MD_ORDINAL = "mo"
    DATE_DM_ORDINAL = "om"
    DATE_Y = "y"


class BreakType(str, Enum):
    """Types of sentence breaks"""

    MINOR = "minor"
    """Break between phrases"""

    MAJOR = "major"
    """Break between sentences"""


class WordRole(str, Enum):
    """Role of a word. Used to disambiguate pronunciations."""

    DEFAULT = ""
    """Use default word pronunciation"""

    LETTER = "gruut:letter"
    """Word should be pronounced as a letter (a = /eɪ/ instead of /ə/)"""


class SSMLParsingState(int, Enum):
    """Current state of SSML parsing"""

    DEFAULT = 0

    IN_WORD = 1
    """Inside <w> or <token>"""

    IN_LEXICON = 2
    """Inside <lexicon>"""

    IN_LEXICON_GRAPHEME = 3
    """Inside <lexicon><grapheme>..."""

    IN_LEXICON_PHONEME = 4
    """Inside <lexicon><phoneme>..."""


@dataclass
class InlineLexicon:
    """SSML lexicon defined inline (not standards compliant)"""

    lexicon_id: str
    alphabet: str = ""

    # word -> role -> [phoneme]
    words: typing.Dict[str, typing.Dict[str, PHONEMES_TYPE]] = field(
        default_factory=dict
    )


@dataclass
class Lexeme:
    """Entry of an inline lexicon"""

    grapheme: str = ""
    phonemes: typing.Optional[PHONEMES_TYPE] = None
    roles: typing.Optional[typing.Set[str]] = None


@dataclass
class Node:
    """Base class of all text processing graph nodes"""

    node: NODE_TYPE
    element: typing.Optional[etree.Element] = None
    voice: str = ""
    lang: str = ""
    implicit: bool = False


@dataclass
class IgnoreNode(Node):
    """Node should be ignored"""

    pass


@dataclass
class BreakNode(Node):
    """Represents a user-specified break"""

    time: str = ""
    """Length of break in seconds (123s) or milliseconds (123ms)"""

    def get_milliseconds(self) -> int:
        """Get number of milliseconds from the time string"""
        if self.time.endswith("ms"):
            return int(self.time[:-2])

        if self.time.endswith("s"):
            return int(float(self.time[:-1]) * 1000)

        return 0


@dataclass
class MarkNode(Node):
    """Represents a user-specified mark"""

    name: str = ""
    """Name of the mark"""


@dataclass
class WordNode(Node):
    """Represents a single word"""

    text: str = ""
    text_with_ws: str = ""
    interpret_as: typing.Union[str, InterpretAs] = ""
    format: typing.Union[str, InterpretAsFormat] = ""

    number: typing.Optional[Decimal] = None
    date: typing.Optional[datetime] = None
    currency_symbol: typing.Optional[str] = None
    currency_name: typing.Optional[str] = None
    time: typing.Optional[Time] = None

    role: typing.Union[str, WordRole] = WordRole.DEFAULT
    pos: typing.Optional[str] = None
    phonemes: typing.Optional[typing.Sequence[str]] = None

    in_lexicon: typing.Optional[bool] = None
    lexicon_ids: typing.Optional[typing.Sequence[str]] = None

    # Assume yes until proven otherwise
    is_maybe_number: bool = True
    is_maybe_date: bool = True
    is_maybe_currency: bool = True
    is_maybe_time: bool = True

    is_from_broken_word: bool = False


@dataclass
class BreakWordNode(Node):
    """Represents a major/minor break in the text"""

    break_type: typing.Union[str, BreakType] = ""
    text: str = ""
    text_with_ws: str = ""


@dataclass
class PunctuationWordNode(Node):
    """Represents a punctuation marker in the text"""

    text: str = ""
    text_with_ws: str = ""


@dataclass
class SentenceNode(Node):
    """Represents a sentence with WordNodes under it"""

    pass


@dataclass
class ParagraphNode(Node):
    """Represents a paragraph with SentenceNodes under it"""

    pass


@dataclass
class SpeakNode(Node):
    """Top-level node for SSML"""

    pass


# -----------------------------------------------------------------------------


@dataclass
class Word:
    """Processed word from a Sentence"""

    idx: int
    """Zero-based index of word in sentence"""

    text: str
    """Text with normalized whitespace"""

    text_with_ws: str
    """Text with original whitespace"""

    leading_ws: str = ""
    """Whitespace before text"""

    trailing_ws: str = ""
    """Whitespace after text"""

    sent_idx: int = 0
    """Zero-based index of sentence in paragraph"""

    par_idx: int = 0
    """Zero-based index of paragraph in document"""

    lang: str = ""
    """Language code"""

    voice: str = ""
    """Voice (from SSML)"""

    pos: typing.Optional[str] = None
    """Part of speech (None if not set)"""

    phonemes: typing.Optional[typing.Sequence[str]] = None
    """List of phonemes (None if not set)"""

    is_major_break: bool = False
    """True if word is a major break (separates sentences)"""

    is_minor_break: bool = False
    """True if word is a minor break (separates phrases)"""

    is_punctuation: bool = False
    """True if word is punctuation that surrounds a spoken word (quotes, etc.)"""

    is_break: typing.Optional[bool] = None
    """True if major or minor break"""

    is_spoken: typing.Optional[bool] = None
    """True if word is something that would be spoken during reading (not punctuation or break)"""

    pause_before_ms: int = 0
    """Milliseconds to pause before this word"""

    pause_after_ms: int = 0
    """Milliseconds to pause after this word"""

    marks_before: typing.Optional[typing.List[str]] = None
    """User-defined marks that occur before this word"""

    marks_after: typing.Optional[typing.List[str]] = None
    """User-defined marks that occur after this word"""

    def __post_init__(self):
        if self.is_break is None:
            self.is_break = self.is_major_break or self.is_minor_break

        if self.is_spoken is None:
            self.is_spoken = not (self.is_punctuation or self.is_break)

        self.leading_ws, self.trailing_ws = default_get_whitespace(self.text_with_ws)


@dataclass
class Sentence:
    """Processed sentence from a document"""

    idx: int
    """Zero-based index of sentence in paragraph"""

    text: str
    """Text with normalized whitespace"""

    text_with_ws: str
    """Text with original whitespace"""

    text_spoken: str
    """Text with only spoken words and normalized whitespace"""

    par_idx: int = 0
    """Zero-based index of paragraph in document"""

    lang: str = ""
    """Language code"""

    voice: str = ""
    """Voice (from SSML)"""

    words: typing.List[Word] = field(default_factory=list)
    """Words in the sentence"""

    pause_before_ms: int = 0
    """Milliseconds to pause before this sentence"""

    pause_after_ms: int = 0
    """Milliseconds to pause after this sentence"""

    marks_before: typing.Optional[typing.List[str]] = None
    """User-defined marks that occur before this sentence"""

    marks_after: typing.Optional[typing.List[str]] = None
    """User-defined marks that occur after this sentence"""

    def __iter__(self):
        """Iterates over words"""
        return iter(self.words)

    def __len__(self):
        """Number of words"""
        return len(self.words)

    def __getitem__(self, key):
        """Gets word by index"""
        return self.words[key]


# -----------------------------------------------------------------------------


class LookupPhonemes:
    """Look up phonemes for word/role in a lexicon"""

    def __call__(
        self, word: str, role: typing.Optional[str] = None, do_transforms: bool = True
    ) -> typing.Optional[PHONEMES_TYPE]:
        pass


class GuessPhonemes:
    """Guess phonemes for word/role"""

    def __call__(
        self, word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        pass


class GetPartsOfSpeech:
    """Get part of speech tags for words"""

    def __call__(self, words: typing.Sequence[str]) -> typing.Sequence[str]:
        pass


class PostProcessSentence:
    """Post-process each sentence node after tokenization/phonemization"""

    def __call__(
        self, graph: GraphType, sentence_node: SentenceNode, settings: typing.Any,
    ):
        pass


@dataclass
class EndElement:
    """Wrapper for end of an XML element (used in TextProcessor)"""

    element: etree.Element


# -----------------------------------------------------------------------------


def has_digit(s: str) -> bool:
    """True if string contains at least one digit"""
    return HAS_DIGIT_PATTERN.search(s) is not None


DEFAULT_WORD_PATTERN = re.compile(r"(\s*\S+(?:\s+|$))")


def default_split_words(s: str) -> typing.Iterable[str]:
    """Split text on whitespace"""
    yield from filter(None, DEFAULT_WORD_PATTERN.findall(s))


def default_get_whitespace(s: str) -> typing.Tuple[str, str]:
    """Returns leading and trailing whitespace of a string"""
    leading_ws, trailing_ws = "", ""
    match = SURROUNDING_WHITESPACE_PATTERN.match(s)
    if match is not None:
        leading_ws, trailing_ws = match.groups()

    return leading_ws, trailing_ws


def default_normalize_whitespace(s: str) -> str:
    """Replace multiple spaces with single space"""
    return NORMALIZE_WHITESPACE_PATTERN.sub(" ", s.strip())


def maybe_compile_regex(
    str_or_pattern: typing.Union[str, REGEX_PATTERN]
) -> REGEX_PATTERN:
    """Compile regex pattern if it's a string"""
    if isinstance(str_or_pattern, REGEX_PATTERN):
        return str_or_pattern

    assert isinstance(str_or_pattern, str)

    return re.compile(str_or_pattern)


# -----------------------------------------------------------------------------


@dataclass
class TextProcessorSettings:
    """Language specific settings for text processing"""

    lang: str
    """Language code that these settings apply to (e.g., en_US)"""

    # Whitespace/tokenization
    split_words: typing.Callable[[str], typing.Iterable[str]] = default_split_words
    """Split text into words and separators"""

    join_str: str = " "
    """String used to combine text from words"""

    keep_whitespace: bool = True
    """True if original whitespace should be retained"""

    is_non_word: typing.Optional[typing.Callable[[str], bool]] = None
    """Returns true if text is not a word (and should be ignored in final output)"""

    get_whitespace: typing.Callable[
        [str], typing.Tuple[str, str]
    ] = default_get_whitespace
    """Returns leading, trailing whitespace from a string"""

    normalize_whitespace: typing.Callable[[str], str] = default_normalize_whitespace
    """Normalizes whitespace in a string"""

    # Punctuations
    begin_punctuations: typing.Optional[typing.Set[str]] = None
    """Strings that should be split off from the beginning of a word"""

    begin_punctuations_pattern: typing.Optional[REGEX_TYPE] = None
    """Regex that overrides begin_punctuations"""

    end_punctuations: typing.Optional[typing.Set[str]] = None
    """Strings that should be split off from the end of a word"""

    end_punctuations_pattern: typing.Optional[REGEX_TYPE] = None
    """Regex that overrides end_punctuations"""

    # Replacements/abbreviations
    replacements: typing.Sequence[typing.Tuple[REGEX_TYPE, str]] = field(
        default_factory=list
    )
    """Regex, replacement template pairs that are applied in order right after tokenization on each word"""

    abbreviations: typing.Dict[REGEX_TYPE, str] = field(default_factory=dict)
    """Regex, replacement template pairs that may expand words after minor breaks are matched"""

    spell_out_words: typing.Dict[str, str] = field(default_factory=dict)
    """Written form, spoken form pairs that are applied with interpret-as="spell-out" in <say-as>"""

    # Breaks
    major_breaks: typing.Set[str] = field(default_factory=set)
    """Set of strings that occur at the end of a word and should break apart sentences."""

    major_breaks_pattern: typing.Optional[REGEX_TYPE] = None
    """Regex that overrides major_breaks"""

    minor_breaks: typing.Set[str] = field(default_factory=set)
    """Set of strings that occur at the end of a word and should break apart phrases."""

    minor_breaks_pattern: typing.Optional[REGEX_TYPE] = None
    """Regex that overrides minor_breaks"""

    word_breaks: typing.Set[str] = field(default_factory=set)
    word_breaks_pattern: typing.Optional[REGEX_TYPE] = None
    """Regex that overrides word_breaks"""

    # Numbers
    is_maybe_number: typing.Optional[typing.Callable[[str], bool]] = has_digit
    """True if a word may be a number (parsing will be attempted)"""

    get_ordinal: typing.Optional[typing.Callable[[str], typing.Optional[int]]] = None
    """Returns integer value of an ordinal string (e.g., 1st -> 1) or None if not an ordinal"""

    babel_locale: typing.Optional[str] = None
    """Locale used to parse numbers/dates/currencies (defaults to lang)"""

    num2words_lang: typing.Optional[str] = None
    """Language used to verbalize numbers (defaults to lang)"""

    # Currency
    default_currency: str = "USD"
    """Currency name to use when interpret-as="currency" but no currency symbol is present"""

    currencies: typing.MutableMapping[str, str] = field(default_factory=dict)
    """Mapping from currency symbol ($) to currency name (USD)"""

    currency_symbols: typing.Sequence[str] = field(default_factory=list)
    """Ordered list of currency symbols (decreasing length)"""

    is_maybe_currency: typing.Optional[typing.Callable[[str], bool]] = has_digit
    """True if a word may be an amount of currency (parsing will be attempted)"""

    # Dates
    dateparser_lang: typing.Optional[str] = None
    """Language used to parse dates (defaults to lang)"""

    is_maybe_date: typing.Optional[typing.Callable[[str], bool]] = has_digit
    """True if a word may be a date (parsing will be attempted)"""

    default_date_format: typing.Union[
        str, InterpretAsFormat
    ] = InterpretAsFormat.DATE_MDY_ORDINAL
    """Format used to verbalize a date unless set with the format attribute of <say-as>"""

    # Times
    is_maybe_time: typing.Optional[typing.Callable[[str], bool]] = has_digit
    """True if a word may be a clock time (parsing will be attempted)"""

    parse_time: typing.Optional[typing.Callable[[str], typing.Optional[Time]]] = None
    """Parse word text into a Time object or None"""

    verbalize_time: typing.Optional[
        typing.Callable[[Time], typing.Iterable[str]]
    ] = None
    """Convert Time to words"""

    # Part of speech (pos) tagging
    get_parts_of_speech: typing.Optional[GetPartsOfSpeech] = None
    """Optional function to get part of speech for a word"""

    # Initialisms (e.g, TTS or T.T.S.)
    is_initialism: typing.Optional[typing.Callable[[str], bool]] = None
    """True if a word is an initialism (will be split with split_initialism)"""

    split_initialism: typing.Optional[
        typing.Callable[[str], typing.Sequence[str]]
    ] = None
    """Function to break apart an initialism into multiple words (called if is_initialism is True)"""

    # Phonemization
    lookup_phonemes: typing.Optional[LookupPhonemes] = None
    """Optional function to look up phonemes for a word/role (without guessing)"""

    guess_phonemes: typing.Optional[GuessPhonemes] = None
    """Optional function to guess phonemes for a word/role"""

    # Pre/post-processing
    pre_process_text: typing.Optional[typing.Callable[[str], str]] = None
    """Optional function to process text during tokenization"""

    post_process_sentence: typing.Optional[PostProcessSentence] = None
    """Optional function to post-process each sentence in the graph before post_process_graph"""

    def __post_init__(self):
        # Languages/locales
        if self.babel_locale is None:
            if "-" in self.lang:
                # en-us -> en_US
                lang_parts = self.lang.split("-", maxsplit=1)
                self.babel_locale = "_".join(
                    [lang_parts[0].lower(), lang_parts[1].upper()]
                )
            else:
                self.babel_locale = self.lang

        if self.num2words_lang is None:
            self.num2words_lang = self.babel_locale

        if self.dateparser_lang is None:
            # en_US -> en
            self.dateparser_lang = self.babel_locale.split("_")[0]

        # Pre-compiled regular expressions
        self.replacements = [
            (maybe_compile_regex(pattern), template)
            for pattern, template in self.replacements
        ]

        compiled_abbreviations = {}
        for pattern, template in self.abbreviations.items():
            if isinstance(pattern, str):
                if not pattern.endswith("$") and self.major_breaks:
                    # Automatically add optional major break at the end
                    break_pattern_str = "|".join(
                        re.escape(b) for b in self.major_breaks
                    )
                    pattern = (
                        f"{pattern}(?P<break>{break_pattern_str})?(?P<whitespace>\\s*)$"
                    )
                    template += r"\g<break>\g<whitespace>"

                pattern = re.compile(pattern)

            compiled_abbreviations[pattern] = template

        self.abbreviations = compiled_abbreviations

        # Strings that should be separated from words, but do not cause any breaks
        if (self.begin_punctuations_pattern is None) and self.begin_punctuations:
            pattern_str = "|".join(re.escape(b) for b in self.begin_punctuations)

            # Match begin_punctuations only at start a word
            self.begin_punctuations_pattern = f"^({pattern_str})"

        if self.begin_punctuations_pattern is not None:
            self.begin_punctuations_pattern = maybe_compile_regex(
                self.begin_punctuations_pattern
            )

        if (self.end_punctuations_pattern is None) and self.end_punctuations:
            pattern_str = "|".join(re.escape(b) for b in self.end_punctuations)

            # Match end_punctuations only at end of a word
            self.end_punctuations_pattern = f"({pattern_str})$"

        if self.end_punctuations_pattern is not None:
            self.end_punctuations_pattern = maybe_compile_regex(
                self.end_punctuations_pattern
            )

        # Major breaks (split sentences)
        if (self.major_breaks_pattern is None) and self.major_breaks:
            pattern_str = "|".join(re.escape(b) for b in self.major_breaks)

            # Match major break with either whitespace at the end or at the end of the text
            # Allow for multiple punctuation symbols (e.g., !?)
            self.major_breaks_pattern = f"((?:{pattern_str})+(?:\\s+|$))"

        if self.major_breaks_pattern is not None:
            self.major_breaks_pattern = maybe_compile_regex(self.major_breaks_pattern)

        # Minor breaks (don't split sentences)
        if (self.minor_breaks_pattern is None) and self.minor_breaks:
            pattern_str = "|".join(re.escape(b) for b in self.minor_breaks)

            # Match minor break with either whitespace at the end or at the end of the text
            self.minor_breaks_pattern = f"((?:{pattern_str})(?:\\s+|$))"

        if self.minor_breaks_pattern is not None:
            self.minor_breaks_pattern = maybe_compile_regex(self.minor_breaks_pattern)

        # Word breaks (break words apart into multiple words)
        if (self.word_breaks_pattern is None) and self.word_breaks:
            pattern_str = "|".join(re.escape(b) for b in self.word_breaks)
            self.word_breaks_pattern = f"(?:{pattern_str})"

        if self.word_breaks_pattern is not None:
            self.word_breaks_pattern = maybe_compile_regex(self.word_breaks_pattern)

        # Currency
        if not self.currencies:
            try:
                # Look up currencies for locale
                locale_obj = babel.Locale(self.babel_locale)

                # $ -> USD
                self.currencies = {
                    babel.numbers.get_currency_symbol(cn): cn
                    for cn in locale_obj.currency_symbols
                }
            except Exception:
                # No automatic currencies
                pass

        if not self.currency_symbols:
            # Currency symbols (e.g., "$") by decreasing length
            self.currency_symbols = sorted(
                self.currencies, key=operator.length_hint, reverse=True
            )


# -----------------------------------------------------------------------------
