"""Shared classes, types, and enums"""
import itertools
import re
import typing
import xml.etree.ElementTree as etree
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

# alias -> full language name
LANG_ALIASES = {
    "ar": "ar",
    "cs": "cs-cz",
    "de": "de-de",
    "en": "en-us",
    "es": "es-es",
    "fa": "fa",
    "fr": "fr-fr",
    "it": "it-it",
    "nl": "nl",
    "pt-br": "pt",
    "ru": "ru-ru",
    "sv": "sv-se",
    "sw": "sw",
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

NODE_TYPE = int
DATA_PROP = "data"


class GraphType(typing.Protocol):
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

    role: typing.Union[str, WordRole] = WordRole.DEFAULT
    pos: typing.Optional[str] = None
    phonemes: typing.Optional[typing.Sequence[str]] = None


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
    text: str
    text_with_ws: str
    sent_idx: int
    lang: str = ""
    pos: typing.Optional[str] = None
    phonemes: typing.Optional[typing.Sequence[str]] = None
    is_break: bool = False
    is_punctuation: bool = False

    @property
    def is_spoken(self):
        """True if word is something that would be spoken during reading"""
        return not (self.is_break or self.is_punctuation)


@dataclass
class Sentence:
    """Processed sentence from a document"""

    idx: int
    text: str
    text_with_ws: str
    lang: str = ""
    voice: str = ""
    words: typing.Sequence[Word] = field(default_factory=list)

    def __iter__(self):
        return iter(self.words)

    def __len__(self):
        return len(self.words)

    def __getitem__(self, key):
        return self.words[key]


# -----------------------------------------------------------------------------


class LookupPhonemes(typing.Protocol):
    """Look up phonemes for word/role in a lexicon"""

    def __call__(
        self, word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        pass


class GuessPhonemes(typing.Protocol):
    """Guess phonemes for word/role"""

    def __call__(
        self, word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        pass


class GetPartsOfSpeech(typing.Protocol):
    """Get part of speech tags for words"""

    def __call__(self, words: typing.Sequence[str]) -> typing.Sequence[str]:
        pass


class PostProcessSentence(typing.Protocol):
    """Post-process each sentence node after tokenization/phonemization"""

    def __call__(
        self, graph: GraphType, sentence_node: SentenceNode, settings: typing.Any,
    ):
        pass


@dataclass
class EndElement:
    """Wrapper for end of an XML element (used in TextProcessor)"""

    element: etree.Element
