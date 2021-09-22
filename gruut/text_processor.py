#!/usr/bin/env python3
import logging
import operator
import re
import sys
import typing
import xml.etree.ElementTree as etree
import xml.sax.saxutils as saxutils
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum

from gruut_ipa import IPA
import babel
import babel.numbers
import dateparser
import networkx as nx
from num2words import num2words

from gruut.const import REGEX_PATTERN, REGEX_TYPE
from gruut.utils import (
    attrib_no_namespace,
    grouper,
    maybe_compile_regex,
    tag_no_namespace,
)

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut.text_processor")

DEFAULT_SPLIT_PATTERN = re.compile(r"(\s+)")
DEFAULT_NON_WORD_PATTERN = re.compile(r"^(\W|_)+$")

NORMALIZE_WHITESPACE_PATTERN = re.compile(r"\s+")
SURROUNDING_WHITESPACE_PATTERN = re.compile(r"^(\s*)\S+(\s*)$")
HAS_DIGIT_PATTERN = re.compile(r"[0-9]")


GRAPH_TYPE = typing.Type[nx.DiGraph]
NODE_TYPE = int
DATA_PROP = "data"
PHONEMES_TYPE = typing.Sequence[str]


class InterpretAs(str, Enum):
    SPELL_OUT = "spell-out"
    """Word should be spelled out (abc = a b c)"""

    DATE = "date"
    """Word should be interpreted as a date"""

    NUMBER = "number"
    """Word should be interpreted as a number"""

    CURRENCY = "currency"
    """Word should be interpreted as an amount of currency"""


class InterpretAsFormat(str, Enum):
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
    MINOR = "minor"
    MAJOR = "major"


class WordRole(str, Enum):
    DEFAULT = ""

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
class BreakNode(Node):
    """Represents a user-specified break"""

    time: str = ""


# TODO: Implement <sub>
@dataclass
class SubNode(Node):
    alias: str = ""


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
    idx: int
    text: str
    text_with_ws: str
    sent_idx: int
    lang: str = ""
    pos: typing.Optional[str] = None
    phonemes: typing.Optional[typing.Sequence[str]] = None
    is_break: bool = False
    is_punctuation: bool = False


@dataclass
class Sentence:
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


def get_whitespace(s: str) -> typing.Tuple[str, str]:
    """Returns leading and trailing whitespace of a string"""
    leading_ws, trailing_ws = "", ""
    match = SURROUNDING_WHITESPACE_PATTERN.match(s)
    if match is not None:
        leading_ws, trailing_ws = match.groups()

    return leading_ws, trailing_ws


def has_digit(s: str) -> bool:
    """True if string contains at least one digit"""
    return HAS_DIGIT_PATTERN.search(s) is not None


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
        self,
        graph: GRAPH_TYPE,
        sentence_node: SentenceNode,
        settings: "TextProcessorSettings",
    ):
        pass


@dataclass
class TextProcessorSettings:
    lang: str

    # Whitespace/tokenization
    split_pattern: REGEX_TYPE = DEFAULT_SPLIT_PATTERN
    join_str: str = " "
    keep_whitespace: bool = True

    # Punctuations
    begin_punctuations: typing.Optional[typing.Set[str]] = None
    begin_punctuations_pattern: typing.Optional[REGEX_TYPE] = None
    end_punctuations: typing.Optional[typing.Set[str]] = None
    end_punctuations_pattern: typing.Optional[REGEX_TYPE] = None

    # Replacements/abbreviations
    replacements: typing.Sequence[typing.Tuple[REGEX_TYPE, str]] = field(
        default_factory=list
    )
    abbreviations: typing.Dict[REGEX_TYPE, str] = field(default_factory=dict)
    spell_out_words: typing.Dict[str, str] = field(default_factory=dict)

    # Breaks
    major_breaks: typing.Set[str] = field(default_factory=set)
    major_breaks_pattern: typing.Optional[REGEX_TYPE] = None
    minor_breaks: typing.Set[str] = field(default_factory=set)
    minor_breaks_pattern: typing.Optional[REGEX_TYPE] = None
    word_breaks: typing.Set[str] = field(default_factory=set)
    word_breaks_pattern: typing.Optional[REGEX_TYPE] = None

    # Numbers
    is_maybe_number: typing.Optional[typing.Callable[[str], bool]] = has_digit
    babel_locale: typing.Optional[str] = None
    num2words_lang: typing.Optional[str] = None

    # Currency
    default_currency: str = "USD"
    currencies: typing.MutableMapping[str, str] = field(default_factory=dict)
    currency_symbols: typing.Sequence[str] = field(default_factory=list)
    is_maybe_currency: typing.Optional[typing.Callable[[str], bool]] = has_digit

    # Dates
    dateparser_lang: typing.Optional[str] = None
    is_maybe_date: typing.Optional[typing.Callable[[str], bool]] = has_digit
    default_date_format: typing.Union[
        str, InterpretAsFormat
    ] = InterpretAsFormat.DATE_MDY_ORDINAL

    # Part of speech (pos) tagging
    get_parts_of_speech: typing.Optional[GetPartsOfSpeech] = None

    # Initialisms (e.g, TTS or T.T.S.)
    is_initialism: typing.Optional[typing.Callable[[str], bool]] = None
    split_initialism: typing.Optional[
        typing.Callable[[str], typing.Sequence[str]]
    ] = None

    # Phonemization
    lookup_phonemes: typing.Optional[LookupPhonemes] = None
    guess_phonemes: typing.Optional[GuessPhonemes] = None

    # Post-processing
    post_process_sentence: typing.Optional[PostProcessSentence] = None

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

        # Pattern to split text into initial words
        self.split_pattern = maybe_compile_regex(self.split_pattern)

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
            self.major_breaks_pattern = f"((?:{pattern_str})(?:\\s+|$))"

        if self.major_breaks_pattern is not None:
            self.major_breaks_pattern = maybe_compile_regex(self.major_breaks_pattern)

        # Minor breaks (don't split sentences)
        if (self.minor_breaks_pattern is None) and self.minor_breaks:
            pattern_str = "|".join(re.escape(b) for b in self.minor_breaks)

            # Match minor break with optional whitespace after
            self.minor_breaks_pattern = f"((?:{pattern_str})\\s*)"

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
            # Look up currencies for locale
            locale_obj = babel.Locale(self.babel_locale)

            # $ -> USD
            self.currencies = {
                babel.numbers.get_currency_symbol(cn): cn
                for cn in locale_obj.currency_symbols
            }

        if not self.currency_symbols:
            # Currency symbols (e.g., "$") by decreasing length
            self.currency_symbols = sorted(
                self.currencies, key=operator.length_hint, reverse=True
            )

    def normalize_whitespace(self, s: str) -> str:
        return NORMALIZE_WHITESPACE_PATTERN.sub(self.join_str, s.strip())


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


class TextProcessor:
    @dataclass
    class EndElement:
        element: etree.Element

    def __init__(
        self,
        default_lang: str = "en_US",
        settings: typing.Optional[
            typing.MutableMapping[str, TextProcessorSettings]
        ] = None,
        **kwargs,
    ):
        self.default_lang = default_lang
        self.default_settings_kwargs = kwargs

        if not settings:
            settings = {}

        if self.default_lang not in settings:
            default_settings = TextProcessorSettings(lang=self.default_lang, **kwargs)
            settings[self.default_lang] = default_settings

        self.settings = settings

    def get_settings(self, lang: typing.Optional[str] = None) -> TextProcessorSettings:
        lang = lang or self.default_lang
        lang_settings = self.settings.get(lang)

        if lang_settings is None:
            _LOGGER.warning(
                "No settings for language %s. Creating default settings.", lang
            )

            # Create default settings for language
            lang_settings = TextProcessorSettings(
                lang=lang, **self.default_settings_kwargs
            )
            self.settings[lang] = lang_settings

        return lang_settings

    def __call__(
        self,
        text: str,
        ssml: bool = False,
        pos: bool = True,
        phonemize: bool = True,
        post_process: bool = True,
        add_speak_tag: bool = True,
    ) -> typing.Tuple[GRAPH_TYPE, Node]:
        if not ssml:
            # Not XML
            text = saxutils.escape(text)

        if add_speak_tag and (not text.lstrip().startswith("<")):
            # Wrap in <speak> tag
            text = f"<speak>{text}</speak>"

        root_element = etree.fromstring(text)
        graph = nx.DiGraph()

        # Parse XML
        last_paragraph: typing.Optional[ParagraphNode] = None
        last_sentence: typing.Optional[SentenceNode] = None
        last_speak: typing.Optional[SpeakNode] = None
        root: typing.Optional[SpeakNode] = None

        # [voice]
        voice_stack: typing.List[str] = []

        # [(interpret_as, format)]
        say_as_stack: typing.List[typing.Tuple[str, str]] = []

        # [(tag, lang)]
        lang_stack: typing.List[typing.Tuple[str, str]] = []
        current_lang: str = self.default_lang

        # True if currently inside <w> or <token>
        in_word: bool = False

        # True if current word is the last one
        is_last_word: bool = False

        # Current word's role
        word_role: typing.Optional[str] = None

        # Create __init__ args for new Node
        def scope_kwargs(target_class):
            scope = {}
            if voice_stack:
                scope["voice"] = voice_stack[-1]

            scope["lang"] = current_lang

            if target_class is WordNode:
                if say_as_stack:
                    scope["interpret_as"], scope["format"] = say_as_stack[-1]

                if word_role is not None:
                    scope["role"] = word_role

            return scope

        # Process sub-elements and text chunks
        for elem_or_text in TextProcessor.text_and_elements(root_element):
            # print(elem_or_text, file=sys.stderr)
            if isinstance(elem_or_text, str):
                # Text chunk
                text = typing.cast(str, elem_or_text)

                if last_speak is None:
                    # Implicit <speak>
                    last_speak = SpeakNode(node=len(graph), implicit=True)
                    graph.add_node(last_speak.node, data=last_speak)
                    if root is None:
                        root = last_speak

                assert last_speak is not None

                if last_paragraph is None:
                    # Implicit <p>
                    p_node = ParagraphNode(
                        node=len(graph), implicit=True, **scope_kwargs(ParagraphNode)
                    )
                    graph.add_node(p_node.node, data=p_node)

                    graph.add_edge(last_speak.node, p_node.node)
                    last_paragraph = p_node

                assert last_paragraph is not None

                if last_sentence is None:
                    # Implicit <s>
                    s_node = SentenceNode(
                        node=len(graph), implicit=True, **scope_kwargs(SentenceNode)
                    )
                    graph.add_node(s_node.node, data=s_node)

                    graph.add_edge(last_paragraph.node, s_node.node)
                    last_sentence = s_node

                assert last_sentence is not None

                if in_word:
                    # No splitting
                    word_text = text
                    settings = self.get_settings(current_lang)
                    if (
                        settings.keep_whitespace
                        and (not is_last_word)
                        and (not word_text.endswith(settings.join_str))
                    ):
                        word_text += settings.join_str

                    word_node = WordNode(
                        node=len(graph),
                        text=word_text.strip(),
                        text_with_ws=word_text,
                        **scope_kwargs(WordNode),
                    )
                    graph.add_node(word_node.node, data=word_node)
                    graph.add_edge(last_sentence.node, word_node.node)
                else:
                    # Split by whitespace
                    self.pipeline_tokenize(
                        graph, last_sentence, text, scope_kwargs=scope_kwargs(WordNode),
                    )

            elif isinstance(elem_or_text, TextProcessor.EndElement):
                # End of an element (e.g., </s>)
                end_elem = typing.cast(TextProcessor.EndElement, elem_or_text)
                end_tag = tag_no_namespace(end_elem.element.tag)

                if end_tag == "voice":
                    if voice_stack:
                        voice_stack.pop()
                elif end_tag == "say-as":
                    if say_as_stack:
                        say_as_stack.pop()
                else:
                    if lang_stack and (lang_stack[-1][0] == end_tag):
                        lang_stack.pop()

                    if lang_stack:
                        current_lang = lang_stack[-1][1]  # tag, lang
                    else:
                        current_lang = self.default_lang

                    if end_tag in {"w", "token"}:
                        # End of word
                        in_word = False
                        is_last_word = False
                        word_role = None

                    if end_tag == "s":
                        # End of sentence
                        last_sentence = None

                    if end_tag == "p":
                        # End of paragraph
                        last_paragraph = None

                    if end_tag == "speak":
                        # End of speak
                        last_speak = root
            else:
                # Start of an element (e.g., <p>)
                elem, elem_metadata = elem_or_text
                elem = typing.cast(etree.Element, elem)

                # Optional metadata for the element
                elem_metadata = typing.cast(
                    typing.Optional[typing.Dict[str, typing.Any]], elem_metadata
                )

                elem_tag = tag_no_namespace(elem.tag)

                if elem_tag == "speak":
                    # Explicit <speak>
                    speak_node = SpeakNode(node=len(graph), element=elem)
                    if root is None:
                        root = speak_node

                    graph.add_node(speak_node.node, data=root)
                    last_speak = root

                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang
                elif elem_tag == "voice":
                    # Set voice scope
                    voice_name = attrib_no_namespace(elem, "name")
                    voice_stack.append(voice_name)
                elif elem_tag == "p":
                    # Explicit paragraph
                    if last_speak is None:
                        # Implicit <speak>
                        last_speak = SpeakNode(node=len(graph), implicit=True)
                        graph.add_node(last_speak.node, data=last_speak)
                        if root is None:
                            root = last_speak

                    assert last_speak is not None

                    p_node = ParagraphNode(
                        node=len(graph), element=elem, **scope_kwargs(ParagraphNode)
                    )
                    graph.add_node(p_node.node, data=p_node)
                    graph.add_edge(last_speak.node, p_node.node)
                    last_paragraph = p_node

                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang
                elif elem_tag == "s":
                    # Explicit sentence
                    if last_speak is None:
                        # Implicit <speak>
                        last_speak = SpeakNode(node=len(graph), implicit=True)
                        graph.add_node(last_speak.node, data=last_speak)
                        if root is None:
                            root = last_speak

                    assert last_speak is not None

                    if last_paragraph is None:
                        # Implicit paragraph
                        p_node = ParagraphNode(
                            node=len(graph), **scope_kwargs(ParagraphNode)
                        )
                        graph.add_node(p_node.node, data=p_node)

                        graph.add_edge(last_speak.node, p_node.node)
                        last_paragraph = p_node

                    s_node = SentenceNode(node=len(graph), element=elem)
                    graph.add_node(s_node.node, data=s_node)
                    graph.add_edge(last_paragraph.node, s_node.node)
                    last_sentence = s_node

                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang
                elif elem_tag in {"w", "token"}:
                    # Explicit word
                    in_word = True
                    is_last_word = (
                        elem_metadata.get("is_last", False) if elem_metadata else False
                    )
                    maybe_lang = attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang

                    word_role = attrib_no_namespace(elem, "role")
                elif elem_tag == "break":
                    # Break
                    last_target = last_sentence or last_paragraph or last_speak
                    assert last_target is not None
                    break_node = BreakNode(
                        node=len(graph),
                        element=elem,
                        time=attrib_no_namespace(elem, "time", ""),
                    )
                    graph.add_node(break_node.node, data=break_node)
                    graph.add_edge(last_target.node, break_node.node)
                elif elem_tag == "say-as":
                    say_as_stack.append(
                        (
                            attrib_no_namespace(elem, "interpret-as", ""),
                            attrib_no_namespace(elem, "format", ""),
                        )
                    )

        assert root is not None

        # Do replacements before minor/major breaks
        self.pipeline_split(self.split_replacements, graph, root)

        # Split punctuations 1/2 (quotes, etc.) before breaks
        self.pipeline_split(self.split_punctuations, graph, root)

        # Split on minor breaks (commas, etc.)
        self.pipeline_split(self.split_minor_breaks, graph, root)

        # Expand abbrevations before major breaks
        self.pipeline_split(self.split_abbreviations, graph, root)

        # Break apart initialisms 1/2 (e.g., TTS or T.T.S.) before major breaks
        self.pipeline_split(self.transform_initialism, graph, root)

        # Split on major breaks (periods, etc.)
        self.pipeline_split(self.split_major_breaks, graph, root)

        # Split punctuations 2/2 (quotes, etc.) after breaks
        self.pipeline_split(self.split_punctuations, graph, root)

        # Break apart initialisms 2/2 (e.g., TTS. or T.T.S..) after major breaks
        self.pipeline_split(self.transform_initialism, graph, root)

        # Break apart sentences using BreakWordNodes
        self.break_sentences(graph, root)

        # spell-out (e.g., abc -> a b c) before number expansion
        self.pipeline_split(self.split_spell_out, graph, root)

        # Transform text into known classes
        self.pipeline_transform(self.transform_number, graph, root)
        self.pipeline_transform(self.transform_currency, graph, root)
        self.pipeline_transform(self.transform_date, graph, root)

        # Verbalize known classes
        self.pipeline_transform(self.verbalize_number, graph, root)
        self.pipeline_transform(self.verbalize_currency, graph, root)
        self.pipeline_transform(self.verbalize_date, graph, root)

        # Break apart words
        self.pipeline_split(self.break_words, graph, root)

        # Gather words from leaves of the tree, group by sentence
        def process_sentence(words: typing.List[WordNode]):
            if pos:
                pos_settings = self.get_settings(node.lang)
                if pos_settings.get_parts_of_speech is not None:
                    pos_tags = pos_settings.get_parts_of_speech(
                        [word.text for word in words]
                    )
                    for word, pos_tag in zip(words, pos_tags):
                        word.pos = pos_tag

                        if not word.role:
                            word.role = f"gruut:{pos_tag}"

            if phonemize:
                # Add phonemes to word
                for word in words:
                    if word.phonemes:
                        # Word already has phonemes
                        continue

                    phonemize_settings = self.get_settings(word.lang)
                    if phonemize_settings.lookup_phonemes is not None:
                        word.phonemes = phonemize_settings.lookup_phonemes(
                            word.text, word.role
                        )

                    if (not word.phonemes) and (
                        phonemize_settings.guess_phonemes is not None
                    ):
                        word.phonemes = phonemize_settings.guess_phonemes(
                            word.text, word.role
                        )

        # Process tree leaves
        sentence_words: typing.List[WordNode] = []

        for dfs_node in nx.dfs_preorder_nodes(graph, root.node):
            node = graph.nodes[dfs_node][DATA_PROP]
            if isinstance(node, SentenceNode):
                if sentence_words:
                    process_sentence(sentence_words)
                    sentence_words = []
            elif graph.out_degree(dfs_node) == 0:
                if isinstance(node, WordNode):
                    word_node = typing.cast(WordNode, node)
                    sentence_words.append(word_node)

        if sentence_words:
            # Final sentence
            process_sentence(sentence_words)
            sentence_words = []

        if post_process:
            # Post-process sentences
            for dfs_node in nx.dfs_preorder_nodes(graph, root.node):
                node = graph.nodes[dfs_node][DATA_PROP]
                if isinstance(node, SentenceNode):
                    sent_node = typing.cast(SentenceNode, node)
                    sent_settings = self.get_settings(sent_node.lang)
                    if sent_settings.post_process_sentence is not None:
                        sent_settings.post_process_sentence(
                            graph, sent_node, sent_settings
                        )

            # Post process entire graph
            self.post_process_graph(graph, root)

        return graph, root

    def post_process_graph(self, graph: GRAPH_TYPE, root: Node):
        pass

    def break_sentences(self, graph: GRAPH_TYPE, root: Node):
        # Break sentences apart at BreakWordNode(break_type="major") nodes.
        # This involves:
        # 1. Identifying where in the edge list of sentence the break occurs
        # 2. Creating a new sentence next to the existing one in the parent paragraph
        # 3. Moving everything after the break into the new sentence
        for leaf_node in list(self.leaves(graph, root,)):
            if not isinstance(leaf_node, BreakWordNode):
                # Not a break
                continue

            break_word_node = typing.cast(BreakWordNode, leaf_node)
            if break_word_node.break_type != BreakType.MAJOR:
                # Not a major break
                continue

            # Get the path from the break up to the nearest sentence
            parent_node: int = next(iter(graph.predecessors(break_word_node.node)))
            parent: Node = graph.nodes[parent_node][DATA_PROP]
            s_path: typing.List[Node] = [parent]

            while not isinstance(parent, SentenceNode):
                parent_node = next(iter(graph.predecessors(parent_node)))
                parent = graph.nodes[parent_node][DATA_PROP]
                s_path.append(parent)

            # Should at least be [WordNode, SentenceNode]
            assert len(s_path) >= 2
            s_node = s_path[-1]
            assert isinstance(s_node, SentenceNode)

            if not s_node.implicit:
                # Don't break apart explicit sentences
                continue

            # Probably a WordNode
            below_s_node = s_path[-2]

            # Edges after the break will need to be moved to the new sentence
            s_edges = list(graph.out_edges(s_node.node))
            break_edge_idx = s_edges.index((s_node.node, below_s_node.node))

            edges_to_move = s_edges[break_edge_idx + 1 :]
            if not edges_to_move:
                # Final sentence, nothing to move
                continue

            # Locate parent paragraph so we can create a new sentence
            p_node = self.find_parent(graph, s_node, ParagraphNode)
            assert p_node is not None

            # Find the index of the edge between the paragraph and the current sentence
            p_s_edge = (p_node.node, s_node.node)
            p_edges = list(graph.out_edges(p_node.node))
            s_edge_idx = p_edges.index(p_s_edge)

            # Remove existing edges from the paragraph
            graph.remove_edges_from(p_edges)

            # Create a sentence and add an edge to it right after the current sentence
            new_s_node = SentenceNode(node=len(graph), implicit=True)
            graph.add_node(new_s_node.node, data=new_s_node)
            p_edges.insert(s_edge_idx + 1, (p_node.node, new_s_node.node))

            # Insert paragraph edges with new sentence
            graph.add_edges_from(p_edges)

            # Move edges from current sentence to new sentence
            graph.remove_edges_from(edges_to_move)
            graph.add_edges_from([(new_s_node.node, v) for (u, v) in edges_to_move])

    def break_words(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as:
            # Don't interpret words that are spoken for
            return

        if not word.implicit:
            # Don't break explicit words
            return

        settings = self.get_settings(word.lang)
        if settings.word_breaks_pattern is None:
            # No pattern set for this language
            return

        if (settings.lookup_phonemes is not None) and settings.lookup_phonemes(
            word.text
        ):
            # Don't break apart words already in the lexicon
            return

        parts = settings.word_breaks_pattern.split(word.text)
        if len(parts) < 2:
            # Didn't split
            return

        # Preserve whitespace
        first_ws, last_ws = get_whitespace(word.text_with_ws)
        last_part_idx = len(parts) - 1

        for part_idx, part_text in enumerate(parts):
            if settings.keep_whitespace:
                if part_idx == 0:
                    part_text = first_ws + part_text

                if part_idx == last_part_idx:
                    part_text += last_ws
                else:
                    part_text += settings.join_str

            yield WordNode, {
                "text": part_text.strip(),
                "text_with_ws": part_text,
                "implicit": True,
                "lang": word.lang,
            }

    def split_punctuations(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)
        if (settings.begin_punctuations_pattern is None) and (
            settings.end_punctuations_pattern is None
        ):
            # No punctuation patterns
            return

        word_text = word.text
        first_ws, last_ws = get_whitespace(word.text_with_ws)
        has_punctuation = False

        # Punctuations at the beginning of the word
        if settings.begin_punctuations_pattern is not None:
            # Split into begin punctuation and rest of word
            parts = list(
                filter(
                    None,
                    settings.begin_punctuations_pattern.split(word_text, maxsplit=1),
                )
            )

            first_word = True
            while word_text and (len(parts) == 2):
                punct_text, word_text = parts
                if first_word:
                    # Preserve leadingwhitespace
                    punct_text = first_ws + punct_text
                    first_word = False

                has_punctuation = True
                yield PunctuationWordNode, {
                    "text": punct_text.strip(),
                    "text_with_ws": punct_text,
                    "implicit": True,
                    "lang": word.lang,
                }

                parts = list(
                    filter(
                        None,
                        settings.begin_punctuations_pattern.split(
                            word_text, maxsplit=1
                        ),
                    )
                )

        # Punctuations at the end of the word
        end_punctuations: typing.List[str] = []
        if settings.end_punctuations_pattern is not None:
            # Split into rest of word and end punctuation
            parts = list(
                filter(
                    None, settings.end_punctuations_pattern.split(word_text, maxsplit=1)
                )
            )

            while word_text and (len(parts) == 2):
                word_text, punct_text = parts
                has_punctuation = True
                end_punctuations.append(punct_text)
                parts = list(
                    filter(
                        None,
                        settings.end_punctuations_pattern.split(word_text, maxsplit=1),
                    )
                )

        if not has_punctuation:
            # Leave word as-is
            return

        if settings.keep_whitespace and (not end_punctuations):
            # Preserve trailing whitespace
            word_text = word_text + last_ws

        if word_text:
            yield WordNode, {
                "text": word_text.strip(),
                "text_with_ws": word_text,
                "implicit": True,
                "lang": word.lang,
            }

        last_punct_idx = len(end_punctuations) - 1
        for punct_idx, punct_text in enumerate(reversed(end_punctuations)):
            if settings.keep_whitespace and (punct_idx == last_punct_idx):
                # Preserve trailing whitespace
                punct_text += last_ws

            yield PunctuationWordNode, {
                "text": punct_text.strip(),
                "text_with_ws": punct_text,
                "implicit": True,
                "lang": word.lang,
            }

    def split_major_breaks(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)
        if settings.major_breaks_pattern is None:
            # No pattern set for this language
            return

        parts = settings.major_breaks_pattern.split(word.text_with_ws)
        if len(parts) < 2:
            return

        word_part = parts[0]
        yield WordNode, {
            "text": word_part.strip(),
            "text_with_ws": word_part,
            "implicit": True,
            "lang": word.lang,
        }

        break_part = parts[1]
        yield BreakWordNode, {
            "break_type": BreakType.MAJOR,
            "text": break_part.strip(),
            "text_with_ws": break_part,
            "implicit": True,
            "lang": word.lang,
        }

    def split_minor_breaks(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as:
            # Don't interpret words that are spoken for
            return

        settings = self.get_settings(word.lang)
        if settings.minor_breaks_pattern is None:
            # No pattern set for this language
            return

        parts = settings.minor_breaks_pattern.split(word.text_with_ws)
        if len(parts) < 2:
            return

        word_part = parts[0]
        yield WordNode, {
            "text": word_part.strip(),
            "text_with_ws": word_part,
            "implicit": True,
            "lang": word.lang,
        }

        break_part = parts[1]
        yield BreakWordNode, {
            "break_type": BreakType.MINOR,
            "text": break_part.strip(),
            "text_with_ws": break_part,
            "implicit": True,
            "lang": word.lang,
        }

    def find_parent(self, graph, node, *classes):
        parents = []
        for parent_node in graph.predecessors(node.node):
            parent = graph.nodes[parent_node][DATA_PROP]
            if isinstance(parent, classes):
                return parent

            parents.append(parent)

        for parent in parents:
            match = self.find_parent(graph, parent, classes)
            if match is not None:
                return match

        return None

    def words(self, graph: GRAPH_TYPE, root: Node, **kwargs) -> typing.Iterable[Word]:
        for sent in self.sentences(graph, root, **kwargs):
            for word in sent:
                yield word

    def phonemes_for_break(
        self,
        break_type: typing.Union[str, BreakType],
        lang: typing.Optional[str] = None,
    ) -> typing.Optional[PHONEMES_TYPE]:
        if break_type == BreakType.MAJOR:
            return [IPA.BREAK_MAJOR.value]

        if break_type == BreakType.MINOR:
            return [IPA.BREAK_MINOR.value]

        return None

    def sentences(
        self,
        graph: GRAPH_TYPE,
        root: Node,
        major_breaks: bool = True,
        minor_breaks: bool = True,
        punctuations: bool = True,
        explicit_lang: bool = True,
        break_phonemes: bool = True,
    ) -> typing.Iterable[Sentence]:
        def get_lang(lang: str) -> str:
            if explicit_lang or (lang != self.default_lang):
                return lang

            # Implicit default language
            return ""

        def make_sentence(
            node: Node, words: typing.Sequence[Word], sent_idx: int
        ) -> Sentence:
            settings = self.get_settings(node.lang)
            text_with_ws = "".join(w.text_with_ws for w in words)
            text = settings.normalize_whitespace(text_with_ws)

            return Sentence(
                idx=sent_idx,
                text=text,
                text_with_ws=text_with_ws,
                lang=get_lang(node.lang),
                voice=node.voice,
                words=words,
            )

        sent_idx: int = 0
        word_idx: int = 0
        words: typing.List[Word] = []
        last_sentence_node: typing.Optional[Node] = None

        for dfs_node in nx.dfs_preorder_nodes(graph, root.node):
            node = graph.nodes[dfs_node][DATA_PROP]
            if isinstance(node, SentenceNode):
                if words:
                    yield make_sentence(node, words, sent_idx)
                    sent_idx += 1
                    word_idx = 0
                    words = []

                last_sentence_node = node
            elif graph.out_degree(dfs_node) == 0:
                if isinstance(node, WordNode):
                    word = typing.cast(WordNode, node)

                    words.append(
                        Word(
                            idx=word_idx,
                            sent_idx=sent_idx,
                            text=word.text,
                            text_with_ws=word.text_with_ws,
                            phonemes=word.phonemes,
                            pos=word.pos,
                            lang=get_lang(node.lang),
                        )
                    )

                    word_idx += 1
                elif isinstance(node, BreakWordNode):
                    break_word = typing.cast(BreakWordNode, node)
                    if (
                        minor_breaks and (break_word.break_type == BreakType.MINOR)
                    ) or (major_breaks and (break_word.break_type == BreakType.MAJOR)):
                        words.append(
                            Word(
                                idx=word_idx,
                                sent_idx=sent_idx,
                                text=break_word.text,
                                text_with_ws=break_word.text_with_ws,
                                phonemes=self.phonemes_for_break(
                                    break_word.break_type, lang=break_word.lang
                                )
                                if break_phonemes
                                else None,
                                is_break=True,
                                lang=get_lang(node.lang),
                            )
                        )

                        word_idx += 1
                elif punctuations and isinstance(node, PunctuationWordNode):
                    punct_word = typing.cast(PunctuationWordNode, node)
                    words.append(
                        Word(
                            idx=word_idx,
                            sent_idx=sent_idx,
                            text=punct_word.text,
                            text_with_ws=punct_word.text_with_ws,
                            is_punctuation=True,
                            lang=get_lang(punct_word.lang),
                        )
                    )

                    word_idx += 1

        if words and (last_sentence_node is not None):
            yield make_sentence(last_sentence_node, words, sent_idx)

    def leaves(self, graph: GRAPH_TYPE, node: Node):
        """Iterate through the leaves of a graph in depth-first order"""
        for dfs_node in nx.dfs_preorder_nodes(graph, node.node):
            if not graph.out_degree(dfs_node) == 0:
                continue

            yield graph.nodes[dfs_node][DATA_PROP]

    # -------------------------------------------------------------------------

    def pipeline_tokenize(
        self, graph, parent_node, text, scope_kwargs=None,
    ):
        if scope_kwargs is None:
            scope_kwargs = {}

        lang = self.default_lang
        if scope_kwargs is not None:
            lang = scope_kwargs.get("lang", lang)

        settings = self.get_settings(lang)
        assert settings is not None

        # Split into separate words/separators.
        # Drop empty words (leading whitespace is still preserved).
        groups = [g for g in grouper(settings.split_pattern.split(text), 2) if g[0]]

        # Preserve whitespace.
        # NOTE: Trailing whitespace will be included in split separator.
        first_ws, _last_ws = get_whitespace(text)

        for group_idx, group in enumerate(groups):
            part_str, sep_str = group
            sep_str = sep_str or ""
            word_text = part_str

            if settings.keep_whitespace:
                if group_idx == 0:
                    word_text = first_ws + word_text

                word_text += sep_str

            word_node = WordNode(
                node=len(graph),
                text=word_text.strip(),
                text_with_ws=word_text,
                implicit=True,
                **scope_kwargs,
            )
            graph.add_node(word_node.node, data=word_node)
            graph.add_edge(parent_node.node, word_node.node)

    def pipeline_transform(
        self, transform_func, graph: GRAPH_TYPE, parent_node: Node,
    ):
        for leaf_node in list(self.leaves(graph, parent_node)):
            transform_func(graph, leaf_node)

    def pipeline_split(
        self, split_func, graph: GRAPH_TYPE, parent_node: Node,
    ):
        for leaf_node in list(self.leaves(graph, parent_node)):
            for node_class, node_kwargs in split_func(graph, leaf_node):
                new_node = node_class(node=len(graph), **node_kwargs)
                graph.add_node(new_node.node, data=new_node)
                graph.add_edge(leaf_node.node, new_node.node)

    # -------------------------------------------------------------------------

    def split_spell_out(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(Word, node)
        if word.interpret_as != InterpretAs.SPELL_OUT:
            return

        settings = self.get_settings(word.lang)

        # Preserve whitespace
        first_ws, last_ws = get_whitespace(word.text_with_ws)
        last_char_idx = len(word.text) - 1

        for i, c in enumerate(word.text):
            # Look up in settings first ("." -> "dot")
            word_text = settings.spell_out_words.get(c)
            role = WordRole.DEFAULT

            if word_text is None:
                if c.isalpha():
                    # Assume this is a letter
                    word_text = c
                    role = WordRole.LETTER
                else:
                    # Leave as is (expand later in pipeline if digit, etc.)
                    word_text = c

            if not word_text:
                continue

            if settings.keep_whitespace:
                if i == 0:
                    word_text = first_ws + word_text

                if i == last_char_idx:
                    word_text += last_ws
                else:
                    word_text += settings.join_str

            yield WordNode, {
                "text": word_text.strip(),
                "text_with_ws": word_text,
                "implicit": True,
                "lang": word.lang,
                "role": role,
            }

    def split_replacements(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as and (word.interpret_as != InterpretAs.NUMBER):
            return

        settings = self.get_settings(word.lang)

        if not settings.replacements:
            # No replacements
            return

        matched = False
        new_text = word.text_with_ws

        for pattern, template in settings.replacements:
            assert isinstance(pattern, REGEX_PATTERN)
            new_text, num_subs = pattern.subn(template, new_text)

            if num_subs > 0:
                matched = True

        if matched:
            # Tokenize new text
            for part_str, sep_str in grouper(settings.split_pattern.split(new_text), 2):
                if settings.keep_whitespace:
                    part_str += sep_str or ""

                if not part_str.strip():
                    # Ignore empty words
                    continue

                yield WordNode, {
                    "text": settings.normalize_whitespace(part_str),
                    "text_with_ws": part_str,
                    "implicit": True,
                    "lang": word.lang,
                }

    def split_abbreviations(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as and (word.interpret_as != InterpretAs.NUMBER):
            return

        settings = self.get_settings(word.lang)

        if not settings.abbreviations:
            # No abbreviations
            return

        new_text: typing.Optional[str] = None
        for pattern, template in settings.abbreviations.items():
            assert isinstance(pattern, REGEX_PATTERN), pattern
            match = pattern.match(word.text_with_ws)

            if match is not None:
                new_text = match.expand(template)
                break

        if new_text is not None:
            # Tokenize new text
            for part_str, sep_str in grouper(settings.split_pattern.split(new_text), 2):
                if settings.keep_whitespace:
                    part_str += sep_str or ""

                if not part_str.strip():
                    # Ignore empty words
                    continue

                yield WordNode, {
                    "text": settings.normalize_whitespace(part_str),
                    "text_with_ws": part_str,
                    "implicit": True,
                    "lang": word.lang,
                }

    def transform_number(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as and (word.interpret_as != InterpretAs.NUMBER):
            return

        settings = self.get_settings(word.lang)
        assert settings.babel_locale

        try:
            # Try to parse as a number
            # This is important to handle thousand/decimal separators correctly.
            number = babel.numbers.parse_decimal(
                word.text, locale=settings.babel_locale
            )
            word.interpret_as = InterpretAs.NUMBER
            word.number = number
        except ValueError:
            pass

    def transform_currency(
        self, graph: GRAPH_TYPE, node: Node,
    ):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as and (word.interpret_as != InterpretAs.CURRENCY):
            return

        settings = self.get_settings(word.lang)

        if (settings.is_maybe_currency is not None) and not settings.is_maybe_currency(
            word.text
        ):
            # Probably not currency
            return

        assert settings.babel_locale

        # Try to parse with known currency symbols
        parsed = False
        for currency_symbol in settings.currency_symbols:
            if word.text.startswith(currency_symbol):
                num_str = word.text[len(currency_symbol) :]
                try:
                    # Try to parse as a number
                    # This is important to handle thousand/decimal separators correctly.
                    number = babel.numbers.parse_decimal(
                        num_str, locale=settings.babel_locale
                    )
                    word.interpret_as = InterpretAs.CURRENCY
                    word.currency_symbol = currency_symbol
                    word.number = number
                    parsed = True
                    break
                except ValueError:
                    pass

        # If this *must* be a currency value, use the default currency
        if (not parsed) and (word.interpret_as == InterpretAs.CURRENCY):
            default_currency = settings.default_currency
            if default_currency:
                # Forced interpretation using default currency
                try:
                    number = babel.numbers.parse_decimal(
                        word.text, locale=settings.babel_locale
                    )
                    word.interpret_as = InterpretAs.CURRENCY
                    word.currency_name = default_currency
                    word.number = number
                except ValueError:
                    pass

    def transform_date(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as and (word.interpret_as != InterpretAs.DATE):
            return

        settings = self.get_settings(word.lang)

        if (settings.is_maybe_date is not None) and not settings.is_maybe_date(
            word.text
        ):
            # Probably not a date
            return

        assert settings.dateparser_lang

        dateparser_kwargs: typing.Dict[str, typing.Any] = {
            "settings": {"STRICT_PARSING": True},
            "languages": [settings.dateparser_lang],
        }

        date = dateparser.parse(word.text, **dateparser_kwargs)
        if date is not None:
            word.interpret_as = InterpretAs.DATE
            word.date = date
        elif word.interpret_as == InterpretAs.DATE:
            # Try again without strict parsing
            dateparser_kwargs["settings"]["STRICT_PARSING"] = False
            date = dateparser.parse(word.text, **dateparser_kwargs)
            if date is not None:
                word.date = date

    def transform_initialism(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if word.interpret_as:
            return

        settings = self.get_settings(word.lang)

        if (settings.is_initialism is None) or (settings.split_initialism is None):
            # Can't do anything without these functions
            return

        if (settings.lookup_phonemes is not None) and settings.lookup_phonemes(
            word.text
        ):
            # Don't expand words already in lexicon
            return

        if not settings.is_initialism(word.text):
            # Not an initialism
            return

        # Split according to language-specific function
        parts = settings.split_initialism(word.text)
        if not parts:
            return

        # Preserve whitespace
        first_ws, last_ws = get_whitespace(word.text_with_ws)
        last_part_idx = len(parts) - 1

        for part_idx, part_text in enumerate(parts):
            if not part_text:
                continue

            if settings.keep_whitespace:
                if part_idx == 0:
                    part_text = first_ws + part_text

                if part_idx == last_part_idx:
                    part_text += last_ws
                else:
                    part_text += settings.join_str

            yield WordNode, {
                "text": part_text.strip(),
                "text_with_ws": part_text,
                "implicit": True,
                "lang": word.lang,
                "role": WordRole.LETTER,
            }

    # -------------------------------------------------------------------------

    def verbalize_number(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if (word.interpret_as != InterpretAs.NUMBER) or (word.number is None):
            return

        settings = self.get_settings(word.lang)

        if (settings.is_maybe_number is not None) and not settings.is_maybe_number(
            word.text
        ):
            # Probably not a number
            return

        assert settings.num2words_lang
        num2words_kwargs = {"lang": settings.num2words_lang}
        decimal_nums = [word.number]

        if word.format == InterpretAsFormat.NUMBER_CARDINAL:
            num2words_kwargs["to"] = "cardinal"
        elif word.format == InterpretAsFormat.NUMBER_ORDINAL:
            num2words_kwargs["to"] = "ordinal"
        elif word.format == InterpretAsFormat.NUMBER_YEAR:
            num2words_kwargs["to"] = "year"
        elif word.format == InterpretAsFormat.NUMBER_DIGITS:
            num2words_kwargs["to"] = "cardinal"
            decimal_nums = [Decimal(d) for d in str(word.number.to_integral_value())]

        for decimal_num in decimal_nums:
            num_has_frac = (decimal_num % 1) != 0

            # num2words uses the number as an index sometimes, so it *has* to be
            # an integer, unless we're doing currency.
            if num_has_frac:
                final_num = float(decimal_num)
            else:
                final_num = int(decimal_num)

            # Convert to words (e.g., 100 -> one hundred)
            num_str = num2words(final_num, **num2words_kwargs)

            # Remove all non-word characters
            num_str = re.sub(r"\W", settings.join_str, num_str).strip()

            # Split into separate words/separators
            groups = list(grouper(settings.split_pattern.split(num_str), 2))

            # Preserve whitespace
            first_ws, last_ws = get_whitespace(word.text_with_ws)
            last_group_idx = len(groups) - 1

            # Split into separate words/separators
            for group_idx, group in enumerate(groups):
                part_str, sep_str = group
                if not part_str:
                    continue

                sep_str = sep_str or ""
                number_word_text = part_str

                if settings.keep_whitespace:
                    if group_idx == 0:
                        number_word_text = first_ws + number_word_text

                    if group_idx == last_group_idx:
                        number_word_text += last_ws
                    else:
                        number_word_text += sep_str

                number_word = WordNode(
                    node=len(graph),
                    implicit=True,
                    lang=word.lang,
                    text=number_word_text.strip(),
                    text_with_ws=number_word_text,
                )
                graph.add_node(number_word.node, data=number_word)
                graph.add_edge(word.node, number_word.node)

    def verbalize_date(self, graph: GRAPH_TYPE, node: Node):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if (word.interpret_as != InterpretAs.DATE) or (word.date is None):
            return

        settings = self.get_settings(word.lang)
        assert settings.babel_locale
        assert settings.num2words_lang

        date = word.date
        date_format = (word.format or settings.default_date_format).strip().upper()
        day_card_str = ""
        day_ord_str = ""
        month_str = ""
        year_str = ""

        if "M" in date_format:
            month_str = babel.dates.format_date(
                date, "MMMM", locale=settings.babel_locale
            )

        num2words_kwargs = {"lang": settings.num2words_lang}

        if "D" in date_format:
            # Cardinal day (1 -> one)
            num2words_kwargs["to"] = "cardinal"
            day_card_str = num2words(date.day, **num2words_kwargs)

        if "O" in date_format:
            # Ordinal day (1 -> first)
            num2words_kwargs["to"] = "ordinal"
            day_ord_str = num2words(date.day, **num2words_kwargs)

        if "Y" in date_format:
            num2words_kwargs["to"] = "year"
            year_str = num2words(date.year, **num2words_kwargs)

        # Transform into Python format string
        # MDY -> {M} {D} {Y}
        date_format_str = settings.join_str.join(f"{{{c}}}" for c in date_format)
        date_str = date_format_str.format(
            M=month_str, D=day_card_str, O=day_ord_str, Y=year_str
        )

        # Split into separate words/separators
        groups = list(grouper(settings.split_pattern.split(date_str), 2))

        # Preserve whitespace
        first_ws, last_ws = get_whitespace(word.text_with_ws)
        last_group_idx = len(groups) - 1

        for group_idx, group in enumerate(groups):
            part_str, sep_str = group
            if not part_str:
                continue

            sep_str = sep_str or ""
            date_word_text = part_str

            if settings.keep_whitespace:
                if group_idx == 0:
                    date_word_text = first_ws + date_word_text

                if group_idx == last_group_idx:
                    date_word_text += last_ws
                else:
                    date_word_text += sep_str

            if not date_word_text:
                continue

            date_word = WordNode(
                node=len(graph),
                implicit=True,
                lang=word.lang,
                text=date_word_text.strip(),
                text_with_ws=date_word_text,
            )
            graph.add_node(date_word.node, data=date_word)
            graph.add_edge(word.node, date_word.node)

    def verbalize_currency(
        self, graph: GRAPH_TYPE, node: Node,
    ):
        if not isinstance(node, WordNode):
            return

        word = typing.cast(WordNode, node)
        if (
            (word.interpret_as != InterpretAs.CURRENCY)
            or ((word.currency_symbol is None) and (word.currency_name is None))
            or (word.number is None)
        ):
            return

        settings = self.get_settings(word.lang)
        assert settings.num2words_lang

        decimal_num = word.number

        # True if number has non-zero fractional part
        num_has_frac = (decimal_num % 1) != 0

        num2words_kwargs = {"lang": settings.num2words_lang, "to": "currency"}

        # Name of currency (e.g., USD)
        if not word.currency_name:
            currency_name = settings.default_currency
            if settings.currencies:
                # Look up currency in locale
                currency_name = settings.currencies.get(
                    word.currency_symbol, settings.default_currency
                )

            word.currency_name = currency_name

        num2words_kwargs["currency"] = word.currency_name

        # Custom separator so we can remove 'zero cents'
        num2words_kwargs["separator"] = "|"

        try:
            num_str = num2words(float(decimal_num), **num2words_kwargs)
        except Exception:
            _LOGGER.exception("verbalize_currency: %s", word)
            return

        # Post-process currency words
        if num_has_frac:
            # Discard num2words separator
            num_str = num_str.replace("|", "")
        else:
            # Remove 'zero cents' part
            num_str = num_str.split("|", maxsplit=1)[0]

        # Remove all non-word characters
        num_str = re.sub(r"\W", settings.join_str, num_str).strip()

        # Split into separate words/separators
        groups = list(grouper(settings.split_pattern.split(num_str), 2))

        # Preserve whitespace
        first_ws, last_ws = get_whitespace(word.text_with_ws)
        last_group_idx = len(groups) - 1

        # Split into separate words
        for group_idx, group in enumerate(groups):
            part_str, sep_str = group
            if not part_str:
                continue

            sep_str = sep_str or ""
            currency_word_text = part_str

            if settings.keep_whitespace:
                if group_idx == 0:
                    currency_word_text = first_ws + currency_word_text

                if group_idx == last_group_idx:
                    currency_word_text += last_ws
                else:
                    currency_word_text += sep_str

            currency_word = WordNode(
                node=len(graph),
                implicit=True,
                lang=word.lang,
                text=currency_word_text.strip(),
                text_with_ws=currency_word_text,
            )
            graph.add_node(currency_word.node, data=currency_word)
            graph.add_edge(word.node, currency_word.node)

    # -------------------------------------------------------------------------

    @staticmethod
    def text_and_elements(element, is_last=False):
        element_metadata = None

        if is_last:
            # True if this is the last child element of a parent.
            # Used to preserve whitespace.
            element_metadata = {"is_last": True}

        yield element, element_metadata

        # Text before any tags (or end tag)
        text = element.text if element.text is not None else ""
        if text.strip():
            yield text

        children = list(element)
        last_child_idx = len(children) - 1

        for child_idx, child in enumerate(children):
            # Sub-elements
            is_last = child_idx == last_child_idx
            yield from TextProcessor.text_and_elements(child, is_last=is_last)

        # End of current element
        yield TextProcessor.EndElement(element)

        # Text after the current tag
        tail = element.tail if element.tail is not None else ""
        if tail.strip():
            yield tail

    @staticmethod
    def print_graph(
        graph: GRAPH_TYPE,
        node: typing.Tuple[NODE_TYPE, Node],
        indent: str = "--",
        level: int = 1,
        print_func=print,
    ):
        if isinstance(node, Node):
            n_data = node
            graph_node = node.node
        else:
            graph_node = node
            n_data = graph.nodes[graph_node][DATA_PROP]

        print_func(indent * level, graph_node, n_data)
        for succ_node in graph.successors(graph_node):
            TextProcessor.print_graph(
                graph, succ_node, indent=indent, level=level + 1, print_func=print_func
            )
