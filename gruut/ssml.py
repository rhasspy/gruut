#!/usr/bin/env python3
import functools
import logging
import operator
import os
import re
import sqlite3
import sys
import typing
import unittest
import xml.etree.ElementTree as etree
import collections.abc
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

import babel
import babel.numbers
import dateparser
import networkx as nx
from num2words import num2words

from .const import REGEX_MATCH, REGEX_PATTERN, REGEX_TYPE, Token, TokenFeatures
from .g2p import GraphemesToPhonemes
from .g2p_phonetisaurus import PhonetisaurusGraph
from .utils import grouper, maybe_compile_regex

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut")

DEFAULT_MAJOR_BREAKS = set(".?!")
DEFAULT_MINOR_BREAKS = set(",;:")
DEFAULT_SPLIT_PATTERN = re.compile(r"(\s+)")
DEFAULT_NON_WORD_PATTERN = re.compile(r"^(\W|_)+$")

NO_NAMESPACE_PATTERN = re.compile(r"^{[^}]+}")

HAS_DIGIT_PATTERN = re.compile(r"[0-9]")


class InterpretAs(str, Enum):
    SPELL_OUT = "spell-out"
    DATE = "date"
    NUMBER = "number"
    CURRENCY = "currency"


class InterpretAsFormat(str, Enum):
    NUMBER_CARDINAL = "cardinal"
    NUMBER_ORDINAL = "ordinal"
    NUMBER_DIGITS = "digits"
    NUMBER_YEAR = "year"

    DATE_DMY = "dmy"
    DATE_MDY = "mdy"
    DATE_YMD = "ymd"
    DATE_YM = "ym"
    DATE_MY = "my"
    DATE_MD = "md"
    DATE_DM = "dm"
    DATE_Y = "y"


class BreakType(str, Enum):
    MINOR = "minor"
    MAJOR = "major"


# -----------------------------------------------------------------------------

PHONEMES_TYPE = typing.Sequence[str]


class LookupPhonemes(typing.Protocol):
    def __call__(
        word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        pass


class GuessPhonemes(typing.Protocol):
    def __call__(
        word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        pass


class GetPartsOfSpeech(typing.Protocol):
    def __call__(words: typing.Sequence[str]) -> typing.Sequence[str]:
        pass


def has_digit(s: str) -> bool:
    return HAS_DIGIT_PATTERN.search(s) is not None


@dataclass
class TextProcessorSettings:
    lang: str

    # Whitespace/tokenization
    split_pattern: REGEX_TYPE = DEFAULT_SPLIT_PATTERN
    join_str: str = " "
    keep_whitespace: bool = True

    # Breaks
    major_breaks: typing.Set[str] = field(default_factory=set)
    major_breaks_pattern: typing.Optional[REGEX_TYPE] = None
    minor_breaks: typing.Set[str] = field(default_factory=set)
    minor_breaks_pattern: typing.Optional[REGEX_TYPE] = None

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

    # Part of speech (pos) tagging
    get_parts_of_speech: typing.Optional[GetPartsOfSpeech] = None

    # Phonemization
    lookup_phonemes: typing.Optional[LookupPhonemes] = None
    guess_phonemes: typing.Optional[GuessPhonemes] = None

    def __post_init__(self):
        if self.babel_locale is None:
            self.babel_locale = self.lang

        if self.num2words_lang is None:
            self.num2words_lang = self.lang

        if self.dateparser_lang is None:
            self.dateparser_lang = self.lang.split("_")[0]

        self.split_pattern = maybe_compile_regex(self.split_pattern)

        # Major breaks (split sentences)
        if (self.major_breaks_pattern is None) and self.major_breaks:
            pattern_str = "".join(re.escape(b) for b in self.major_breaks)

            # Match major break with either whitespace at the end or at the end of the text
            self.major_breaks_pattern = f"([{pattern_str}](?:\\s+|$))"

        if self.major_breaks_pattern is not None:
            self.major_breaks_pattern = maybe_compile_regex(self.major_breaks_pattern)

        # Minor breaks (don't split sentences)
        if (self.minor_breaks_pattern is None) and self.minor_breaks:
            pattern_str = "".join(re.escape(b) for b in self.minor_breaks)

            # Match minor break with optional whitespace after
            self.minor_breaks_pattern = f"([{pattern_str}]\\s*)"

        if self.minor_breaks_pattern is not None:
            self.minor_breaks_pattern = maybe_compile_regex(self.minor_breaks_pattern)

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


# -----------------------------------------------------------------------------

GRAPH_TYPE = typing.Type[nx.DiGraph]


@dataclass
class Node:
    node: int
    element: typing.Optional[etree.Element] = None
    voice: str = ""
    lang: str = ""
    implicit: bool = False


@dataclass
class BreakNode(Node):
    time: str = ""


# TODO: Add <sub>
@dataclass
class SubNode(Node):
    alias: str = ""


@dataclass
class WordNode(Node):
    text: str = ""
    text_with_ws: str = ""
    interpret_as: typing.Union[str, InterpretAs] = ""
    format: typing.Union[str, InterpretAsFormat] = ""

    number: typing.Optional[Decimal] = None
    date: typing.Optional[datetime] = None
    currency_symbol: typing.Optional[str] = None
    currency_name: typing.Optional[str] = None

    role: str = ""
    phonemes: typing.Optional[typing.Sequence[str]] = None


@dataclass
class BreakWordNode(Node):
    break_type: typing.Union[str, BreakType] = ""
    text: str = ""
    text_with_ws: str = ""


@dataclass
class SentenceNode(Node):
    pass


@dataclass
class ParagraphNode(Node):
    pass


@dataclass
class SpeakNode(Node):
    pass


# -----------------------------------------------------------------------------


@dataclass
class Word:
    text: str
    text_with_ws: str
    idx: int
    sent_idx: int
    phonemes: typing.Optional[typing.Sequence[str]] = None


# -----------------------------------------------------------------------------

ROLE_TO_PHONEMES = typing.Dict[str, typing.Sequence[str]]


class Phonemizer:
    DEFAULT_ROLE: str = ""

    def __init__(
        self,
        lang: str = "en_US",
        lexicon: typing.Optional[
            typing.Dict[str, typing.Dict[str, ROLE_TO_PHONEMES]]
        ] = None,
        db_conn: typing.Optional[typing.Dict[str, sqlite3.Connection]] = None,
        g2p_model: typing.Optional[typing.Dict[str, typing.Union[str, Path]]] = None,
        word_transform_funcs=None,
        guess_transform_func=None,
    ):
        self.default_lang = lang

        # lang -> word -> [phonemes]
        self.lexicon = lexicon if lexicon is not None else {}

        # lang -> connection
        self.db_conn = db_conn or {}

        # lang -> [functions]
        self.word_transform_funcs = word_transform_funcs

        # lang -> function
        self.guess_transform_func = guess_transform_func

        # lang -> g2p
        self.g2p_tagger: typing.Dict[str, GraphemesToPhonemes] = {}
        self.g2p_graph: typing.Dict[PhonetisaurusGraph] = {}

        if g2p_model is not None:
            for g2p_lang, g2p_path in g2p_model.items():
                # Load g2p model
                g2p_ext = os.path.splitext(g2p_path)[1]
                if g2p_ext == ".npz":
                    # Load Phonetisaurus FST as a numpy graph
                    _LOGGER.debug("Loading Phonetisaurus g2p model from %s", g2p_model)
                    self.g2p_graph[g2p_lang] = PhonetisaurusGraph.load(g2p_model)
                else:
                    # Load CRF tagger
                    _LOGGER.debug("Loading CRF g2p model from %s", g2p_model)
                    self.g2p_tagger[g2p_lang] = GraphemesToPhonemes(g2p_model)

    def lookup(
        self, word: str, lang: typing.Optional[str] = None, transform: bool = True
    ) -> typing.Optional[ROLE_TO_PHONEMES]:
        lang = lang or self.default_lang
        lexicon = self.lexicon.get(lang)
        if lexicon is None:
            lexicon = {}
            self.lexicon[lang] = lexicon

        assert lexicon is not None
        role_to_word = lexicon.get(word)

        if role_to_word is not None:
            return role_to_word

        db_conn = self.db_conn.get(lang)
        if db_conn is not None:
            # Load pronunciations from database.
            #
            # Ordered by pronunciation descending because so duplicate roles
            # will be overwritten by earlier pronunciation.
            cursor = db_conn.execute(
                "SELECT role, phonemes FROM word_phonemes WHERE word = ? ORDER BY pron_order DESC",
                (word,),
            )

            for row in cursor:
                if role_to_word is None:
                    # Create new lexicon entry
                    role_to_word = {}
                    self.lexicon[word] = role_to_word

                role, phonemes = row[0], row[1].split()
                role_to_word[role] = phonemes

            if role_to_word is not None:
                # Successfully looked up in the database
                return role_to_word

        if transform:
            word_transform_funcs = self.word_transform_funcs.get(lang)
            if word_transform_funcs is not None:
                # Try transforming word and looking up again (with transforms
                # disabled, of course)
                for transform_func in word_transform_funcs:
                    maybe_word = transform_func(word)
                    maybe_role_to_word = self.lookup(
                        maybe_word, lang=lang, transform=False
                    )
                    if maybe_role_to_word:
                        # Make a copy for this word
                        role_to_word = dict(maybe_role_to_word)
                        lexicon[word] = role_to_word
                        return role_to_word

        return None

    def get_pronunciation(
        self,
        word: str,
        lang: typing.Optional[str] = None,
        role: typing.Optional[str] = None,
        transform: bool = True,
        guess: bool = True,
    ) -> typing.Optional[typing.Sequence[str]]:
        lang = lang or self.default_lang
        role_to_word = self.lookup(word, lang=lang, transform=transform)
        if role_to_word:
            if role:
                # Desired role
                maybe_phonemes = role_to_word.get(role)
                if maybe_phonemes:
                    return maybe_phonemes

            # Default role
            maybe_phonemes = role_to_word.get(Phonemizer.DEFAULT_ROLE)
            if maybe_phonemes:
                return maybe_phonemes

            # Any role
            return next(iter(role_to_word.values()))

        if role_to_word is None:
            # Mark word as missing
            role_to_word = {}
            self.lexicon[word] = role_to_word

        if guess:
            guessed_phonemes = self.guess_pronunciation(word, lang=lang)
            role_to_word[Phonemizer.DEFAULT_ROLE] = guessed_phonemes
            return guessed_phonemes

        return None

    def guess_pronunciation(
        self, word: str, lang: typing.Optional[str] = None,
    ) -> typing.Sequence[str]:
        lang = lang or self.default_lang
        guess_transform_func = self.guess_transform_func.get(lang)

        if guess_transform_func is not None:
            word = guess_transform_func(word)

        g2p_tagger = self.g2p_tagger.get(lang)

        if g2p_tagger is not None:
            # CRF model
            _LOGGER.debug("Guessing pronunciations for %s with CRF", word)
            return g2p_tagger(word)

        g2p_graph = self.g2p_graph.get(lang)
        if g2p_graph:
            # Phonetisaurus FST
            _LOGGER.debug("Guessing pronunciations for %s with Phonetisaurus", word)
            _, _, guessed_phonemes = next(  # type: ignore
                g2p_graph.g2p([word])
            )

            return guessed_phonemes

        return []


# -----------------------------------------------------------------------------

# MAYBE_LANG = typing.Optional[str]
# POS_TAGGER = typing.Callable[[typing.Iterable[str]], typing.Iterable[str]]


class TextProcessor:
    @dataclass
    class EndElement:
        element: etree.Element

    def __init__(
        self,
        default_lang: str = "en_US",
        settings: typing.Optional[typing.Mapping[str, TextProcessorSettings]] = None,
        **kwargs
        # lang: str = "en_US",
        # phonemizer: typing.Optional[Phonemizer] = None,
        # babel_locale: typing.Optional[
        #     typing.Union[str, typing.Mapping[MAYBE_LANG, str]]
        # ] = None,
        # num2words_lang: typing.Optional[
        #     typing.Union[str, typing.Mapping[MAYBE_LANG, str]]
        # ] = None,
        # dateparser_lang: typing.Optional[
        #     typing.Union[str, typing.Mapping[str, str]]
        # ] = None,
        # split_pattern: typing.Optional[
        #     typing.Union[REGEX_TYPE, typing.Mapping[MAYBE_LANG, REGEX_TYPE]]
        # ] = None,
        # join_str: typing.Dict[MAYBE_LANG, str] = None,
        # keep_whitespace: typing.Dict[MAYBE_LANG, bool] = True,
        # replacements: typing.Optional[
        #     typing.Sequence[typing.Tuple[REGEX_TYPE, str]]
        # ] = None,
        # major_breaks: typing.Optional[typing.Dict[str, typing.Set[str]]] = None,
        # minor_breaks: typing.Optional[typing.Dict[str, typing.Set[str]]] = None,
        # default_currency: typing.Optional[typing.Dict[str, str]] = None,
        # currencies: typing.Optional[typing.Dict[str, typing.Dict[str, str]]] = None,
        # pos_tagger: typing.Optional[
        #     typing.Union[POS_TAGGER, typing.Dict[MAYBE_LANG, POS_TAGGER]]
        # ] = None,
    ):
        self.default_lang = default_lang

        if not settings:
            default_settings = TextProcessorSettings(lang=self.default_lang, **kwargs)
            settings = {self.default_lang: default_settings}

        self.settings = settings
        self.default_settings = self.settings[self.default_lang]

        # self.phonemizer = phonemizer

        # # Default locales/languages
        # self.default_lang = lang

        # # Locale for parsing numbers
        # if babel_locale is None:
        #     # Use default language
        #     babel_locale = lang

        # if not isinstance(babel_locale, collections.abc.Mapping):
        #     # Set for all languages
        #     babel_locale = {None: babel_locale}

        # self.babel_locale = babel_locale

        # # Language for verbalizing numbers
        # if num2words_lang is None:
        #     num2words_lang = lang

        # if not isinstance(num2words_lang, collections.abc.Mapping):
        #     # Set for all languages
        #     num2words_lang = {None: num2words_lang}

        # self.num2words_lang = num2words_lang

        # # Language for parsing dates
        # if dateparser_lang is None:
        #     # en_US -> en
        #     dateparser_lang = lang.split("_")[0]

        # if not isinstance(dateparser_lang, collections.abc.Mapping):
        #     dateparser_lang = {None: dateparser_lang}

        # self.dateparser_lang = dateparser_lang

        # # ----------
        # # Whitespace
        # # ----------

        # # True if whitespace should be kept at the end of words
        # if keep_whitespace is None:
        #     keep_whitespace = True

        # if not isinstance(keep_whitespace, collections.abc.Mapping):
        #     keep_whitespace = {None: keep_whitespace}

        # self.keep_whitespace = keep_whitespace

        # # Regular expression used to split apart words
        # if split_pattern is None:
        #     split_pattern = DEFAULT_SPLIT_PATTERN

        # if not isinstance(split_pattern, collections.abc.Mapping):
        #     # Set for all languages
        #     split_pattern = {None: split_pattern}

        # self.split_pattern = {
        #     k: maybe_compile_regex(v) for k, v in split_pattern.items()
        # }

        # # String used to join words
        # if join_str is None:
        #     join_str = " "

        # if not isinstance(join_str, collections.abc.Mapping):
        #     # Set for all languages
        #     join_str = {None: join_str}

        # self.join_str = join_str

        # # ------
        # # Breaks
        # # ------

        # # Strings that cause sentences to break apart
        # if major_breaks is None:
        #     major_breaks = set()

        # if not isinstance(major_breaks, collections.abc.Mapping):
        #     # Set for all languages
        #     major_breaks = {None: major_breaks}

        # self.major_breaks = major_breaks

        # # Pre-compile regex patterns for matching major breaks
        # self.major_break_pattern: typing.Dict[MAYBE_LANG, REGEX_PATTERN] = {}
        # for break_lang, break_set in self.major_breaks.items():
        #     if not break_set:
        #         continue

        #     break_str = "".join(re.escape(b) for b in break_set)

        #     # Match major break with either whitespace at the end or at the end of the text
        #     self.major_break_pattern[break_lang] = re.compile(
        #         f"([{break_str}](?:\\s+|$))"
        #     )

        # # Strings that separate words, but not sentences
        # if minor_breaks is None:
        #     minor_breaks = set()

        # if not isinstance(minor_breaks, collections.abc.Mapping):
        #     # Set for all languages
        #     minor_breaks = {None: minor_breaks}

        # self.minor_breaks = minor_breaks

        # # Pre-compile regex patterns for matching minor breaks
        # self.minor_break_pattern: typing.Dict[MAYBE_LANG, REGEX_PATTERN] = {}
        # for break_lang, break_set in self.minor_breaks.items():
        #     if not break_set:
        #         continue

        #     break_str = "".join(re.escape(b) for b in break_set)

        #     # Match minor break with optional whitespace after
        #     self.minor_break_pattern[break_lang] = re.compile(f"([{break_str}]\\s*)")

        # # --------
        # # Currency
        # # --------

        # if default_currency is None:
        #     default_currency = "USD"

        # if not isinstance(default_currency, collections.abc.Mapping):
        #     # Set for all languages
        #     default_currency = {None: default_currency}

        # self.default_currency = default_currency

        # # lang (en_US) -> symbol ($) -> name (USD)
        # if currencies is None:
        #     currencies = {}

        # if (not currencies) or (
        #     currencies
        #     and not isinstance(next(iter(currencies.values())), collections.abc.Mapping)
        # ):
        #     # Set for all languages
        #     currencies = {None: currencies}

        # self.currencies = currencies

        # # Sorted by decreasing length
        # # lang -> [longest_symbol, symbol, shortest_symbol]
        # self.currency_symbols = {}

        # # TODO: Handle replacements
        # # self.replacements = [
        # #     (maybe_compile_regex(p), r) for p, r in (replacements or [])
        # # ]

        # if pos_tagger is None:
        #     pos_tagger = {}

        # if not isinstance(pos_tagger, collections.abc.Mapping):
        #     pos_tagger = {None: pos_tagger}

        # self.pos_tagger = pos_tagger

    # def pre_tokenize(self, text: str) -> str:
    #     for pattern, replacement in self.replacements:
    #         text = pattern.sub(replacement, text)

    #     return text

    def get_settings(self, lang: str) -> TextProcessorSettings:
        lang_settings = self.settings.get(lang)
        if lang_settings is None:
            # TODO: Warning or failure here?
            # Create default settings for language
            lang_settings = TextProcessorSettings(lang=lang)
            self.settings[lang] = lang_settings

        return lang_settings

    def process(self, text: str, add_speak_tag: bool = True, phonemize: bool = True):
        if add_speak_tag and (not text.lstrip().startswith("<")):
            # Wrap in <speak> tag
            text = f"<speak>{text}</speak>"

        root_element = etree.fromstring(text)
        graph = nx.DiGraph()

        return self.process_etree(root_element, graph, phonemize=phonemize)

    def process_etree(self, root_element: etree.Element, graph, phonemize: bool = True):
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
                    word_node = WordNode(
                        node=len(graph),
                        text=text.strip(),
                        text_with_ws=text,
                        **scope_kwargs(WordNode),
                    )
                    graph.add_node(word_node.node, data=word_node)
                    graph.add_edge(last_sentence.node, word_node.node)
                else:
                    # Split by whitespace
                    self.pipeline_tokenize(
                        graph,
                        last_sentence,
                        text,
                        settings=self.get_settings(current_lang),
                        scope_kwargs={"implicit": True, **scope_kwargs(WordNode)},
                    )

            elif isinstance(elem_or_text, TextProcessor.EndElement):
                # End of an element (e.g., </s>)
                end_elem = typing.cast(TextProcessor.EndElement, elem_or_text)
                end_tag = TextProcessor.tag_no_namespace(end_elem.element.tag)

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
                elem = typing.cast(etree.Element, elem_or_text)
                elem_tag = TextProcessor.tag_no_namespace(elem.tag)

                if elem_tag == "speak":
                    # Explicit <speak>
                    speak_node = SpeakNode(node=len(graph), element=elem)
                    if root is None:
                        root = speak_node

                    graph.add_node(speak_node.node, data=root)
                    last_speak = root

                    maybe_lang = TextProcessor.attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang
                elif elem_tag == "voice":
                    # Set voice scope
                    voice_name = TextProcessor.attrib_no_namespace(elem, "name")
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

                    maybe_lang = TextProcessor.attrib_no_namespace(elem, "lang")
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

                    maybe_lang = TextProcessor.attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang
                elif elem_tag in {"w", "token"}:
                    # Explicit word
                    in_word = True
                    maybe_lang = TextProcessor.attrib_no_namespace(elem, "lang")
                    if maybe_lang:
                        lang_stack.append((elem_tag, maybe_lang))
                        current_lang = maybe_lang

                    word_role = TextProcessor.attrib_no_namespace(elem, "role")
                elif elem_tag == "break":
                    # Break
                    last_target = last_sentence or last_paragraph or last_speak
                    assert last_target is not None
                    break_node = BreakNode(
                        node=len(graph),
                        element=elem,
                        time=TextProcessor.attrib_no_namespace(elem, "time", ""),
                    )
                    graph.add_node(break_node.node, data=break_node)
                    graph.add_edge(last_target.node, break_node.node)
                elif elem_tag == "say-as":
                    say_as_stack.append(
                        (
                            TextProcessor.attrib_no_namespace(elem, "interpret-as", ""),
                            TextProcessor.attrib_no_namespace(elem, "format", ""),
                        )
                    )

        assert root is not None

        # Split on major/minor breaks
        self.pipeline_split(self.split_major_breaks, graph, root)

        # Break apart sentences using BreakWordNodes
        self.break_sentences(graph, root)

        # if self.minor_break_pattern:
        #     self.pipeline_split(
        #         self.split_minor_breaks, root, graph,
        #     )

        # Transform text into known classes
        self.pipeline_transform(self.transform_number, graph, root)
        # self.pipeline_transform(self.transform_date, graph, root)
        # self.pipeline_transform(self.transform_currency, graph, root)
        # self.pipeline_transform(self.transform_initialism, graph, root)

        # Verbalize known classes
        self.pipeline_transform(self.verbalize_number, graph, root)
        # self.pipeline_transform(self.verbalize_date, graph, root)
        # self.pipeline_transform(self.verbalize_currency, graph, root)

        # # spell-out (e.g., 123 -> 1 2 3)
        # self.pipeline_split(self.split_spell_out, graph, root)

        # # Gather words from leaves of the tree, group by sentence
        # sentences = []
        # words = []
        # for n in self.leaves(graph, root, skip_seen=True):
        #     if isinstance(n, WordNode):
        #         words.append(n)
        #     elif isinstance(n, BreakWordNode) and (n.break_type == BreakType.MAJOR):
        #         sentences.append(words)
        #         words = []

        # if words:
        #     sentences.append(words)
        #     words = []

        # for words in sentences:
        #     # First word on the sentence determines the language for part-of-speech tagging
        #     pos_lang = self.default_lang
        #     if words:
        #         pos_lang = words[0].lang or pos_lang

        #     # Add part-of-speech tags
        #     pos_tagger = self.pos_tagger.get(pos_lang)
        #     if pos_tagger is not None:
        #         # Predict tags for entire sentence
        #         pos_tags = pos_tagger([word.text for word in words])
        #         for word, pos in zip(words, pos_tags):
        #             if not word.role:
        #                 word.role = f"gruut:{pos}"

        #     if phonemize and (self.phonemizer is not None):
        #         # Phonemize words
        #         for word in words:
        #             word.phonemes = self.phonemizer.get_pronunciation(
        #                 word.text, lang=(word.lang or self.default_lang), role=word.role,
        #             )

        TextProcessor.print_graph(graph, root.node)

        return graph, root

    @staticmethod
    def tag_no_namespace(tag: str) -> str:
        return NO_NAMESPACE_PATTERN.sub("", tag)

    @staticmethod
    def attrib_no_namespace(
        element, name: str, default: typing.Any = None
    ) -> typing.Any:
        for key, value in element.attrib.items():
            key_no_ns = NO_NAMESPACE_PATTERN.sub("", key)
            if key_no_ns == name:
                return value

        return default

    def break_sentences(self, graph: GRAPH_TYPE, root: Node):
        # Break sentences apart at BreakWordNode(break_type="major") nodes.
        # This involves:
        # 1. Identifying where in the edge list of sentence the break occurs
        # 2. Creating a new sentence next to the existing one in the parent paragraph
        # 3. Moving everything after the break into the new sentence
        for leaf_node in list(self.leaves(graph, root, skip_seen=True)):
            if not isinstance(leaf_node, BreakWordNode):
                # Not a break
                continue

            break_word_node = typing.cast(BreakWordNode, leaf_node)
            if break_word_node.break_type != BreakType.MAJOR:
                # Not a major break
                continue

            # Get the path from the break up to the nearest sentence
            parent_node: int = next(iter(graph.predecessors(break_word_node.node)))
            parent: Node = graph.nodes[parent_node]["data"]
            s_path: typing.List[Node] = [parent]

            while not isinstance(parent, SentenceNode):
                parent_node = next(iter(graph.predecessors(parent_node)))
                parent = graph.nodes[parent_node]["data"]
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
            "lang": word.lang
        }

    # def split_minor_breaks(self, node, graph):
    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if word.interpret_as:
    #         return

    #     assert self.minor_break_pattern is not None
    #     parts = self.minor_break_pattern.split(word.text_with_ws)
    #     if len(parts) < 2:
    #         return

    #     word_part = parts[0]
    #     yield WordNode, {"text": word_part.strip(), "text_with_ws": word_part}

    #     break_part = parts[1]
    #     yield BreakWordNode, {
    #         "break_type": BreakType.MINOR,
    #         "text": break_part.strip(),
    #         "text_with_ws": break_part,
    #     }

    def find_parent(self, graph, node, *classes):
        parents = []
        for parent_node in graph.predecessors(node.node):
            parent = graph.nodes[parent_node]["data"]
            if isinstance(parent, classes):
                return parent

            parents.append(parent)

        for parent in parents:
            match = self.find_parent(graph, parent, classes)
            if match is not None:
                return match

        return None

    def words(self, graph, node):
        word_idx = 0
        last_sent_idx = None
        sent_idxs = {}

        for leaf_node in self.leaves(graph, node, skip_seen=True):
            if not isinstance(leaf_node, WordNode):
                continue

            word = typing.cast(WordNode, leaf_node)
            sentence = self.find_parent(graph, word, SentenceNode)
            assert sentence is not None, "Word does not belong to a sentence"

            # Get or create index for sentence
            sent_idx = sent_idxs.get(sentence.node)
            if sent_idx is None:
                sent_idx = len(sent_idxs)
                sent_idxs[sentence.node] = sent_idx

            if sent_idx != last_sent_idx:
                # Reset word index
                word_idx = 0
                last_sent_idx = sent_idx

            yield Word(
                idx=word_idx,
                sent_idx=sent_idx,
                text=word.text,
                text_with_ws=word.text_with_ws,
                phonemes=word.phonemes,
            )

            word_idx += 1

    def leaves(self, graph, node, skip_seen=False, seen=None):
        if skip_seen and (seen is None):
            seen = set()

        if graph.out_degree(node.node) == 0:
            if not skip_seen or (node.node not in seen):
                yield node

                if seen is not None:
                    seen.add(node.node)
        else:
            for next_node_n in graph.successors(node.node):
                next_node = graph.nodes[next_node_n]["data"]
                yield from self.leaves(graph, next_node, skip_seen=skip_seen, seen=seen)

    def pipeline_tokenize(
        self, graph, parent_node, text, settings=None, scope_kwargs=None
    ):
        if scope_kwargs is None:
            scope_kwargs = {}

        settings = settings or self.default_lang
        assert settings is not None

        for non_ws, ws in grouper(settings.split_pattern.split(text), 2):
            word = non_ws

            if settings.keep_whitespace:
                word += ws or ""

            if not word:
                continue

            word_node = WordNode(
                node=len(graph), text=non_ws, text_with_ws=word, **scope_kwargs
            )
            graph.add_node(word_node.node, data=word_node)
            graph.add_edge(parent_node.node, word_node.node)

    def pipeline_transform(
        self, transform_func, graph: GRAPH_TYPE, parent_node: Node, skip_seen=True
    ):
        for leaf_node in list(self.leaves(graph, parent_node, skip_seen=skip_seen)):
            transform_func(graph, leaf_node)

    def pipeline_split(
        self, split_func, graph: GRAPH_TYPE, parent_node: Node, skip_seen=True
    ):
        for leaf_node in list(self.leaves(graph, parent_node, skip_seen=skip_seen)):
            for node_class, node_kwargs in split_func(graph, leaf_node):
                new_node = node_class(node=len(graph), **node_kwargs)
                graph.add_node(new_node.node, data=new_node)
                graph.add_edge(leaf_node.node, new_node.node)

    # def split_on_regex(self, node, graph, pattern):
    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if word.interpret_as:
    #         return

    #     parts = pattern.split(word.text)
    #     if len(parts) < 2:
    #         return

    #     for part in parts:
    #         if not part:
    #             continue

    #         yield WordNode, {"text": part.strip(), "text_with_ws": part}

    # def split_spell_out(self, node, graph):
    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if word.interpret_as != InterpretAs.SPELL_OUT:
    #         return

    #     for c in word.text:
    #         yield {"text": c, "role": "gruut:letter"}

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

    # def transform_date(self, node, graph, dateparser_lang: typing.Optional[str] = None):
    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if word.interpret_as and (word.interpret_as != InterpretAs.DATE):
    #         return

    #     if not HAS_DIGIT_PATTERN.search(word.text):
    #         # Dates must have at least one digit
    #         return

    #     word_lang = word.lang or self.default_lang
    #     dateparser_lang = (
    #         dateparser_lang
    #         or self.dateparser_lang.get(word_lang)
    #         or self.dateparser_lang.get(None)
    #         or word_lang
    #     )
    #     assert dateparser_lang is not None

    #     dateparser_kwargs = {
    #         "settings": {"STRICT_PARSING": True},
    #         "languages": [dateparser_lang],
    #     }

    #     # TODO: Filter words first (contains digit?)
    #     date = dateparser.parse(word.text, **dateparser_kwargs)
    #     if date is not None:
    #         word.interpret_as = InterpretAs.DATE
    #         word.date = date

    # def transform_currency(
    #     self,
    #     node,
    #     graph,
    #     babel_locale: typing.Optional[str] = None,
    #     default_currency=None,
    #     currency_symbols=None,
    #     currencies=None,
    # ):

    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if word.interpret_as and (word.interpret_as != InterpretAs.CURRENCY):
    #         return

    #     if not HAS_DIGIT_PATTERN.search(word.text):
    #         # Currency must have at least one digit
    #         return

    #     babel_locale = babel_locale or word.lang or self.babel_locale

    #     # Look up currency symbols for current locale (e.g., "$").
    #     # These must be sorted by decreasing length.
    #     currency_symbols = currency_symbols or self.currency_symbols.get(babel_locale)
    #     if currency_symbols is None:
    #         # Get possible currency symbols and sort them
    #         currencies = currencies or self.currencies.get(babel_locale)
    #         if currencies is None:
    #             # Ask Babel for locale-specific currency symbols
    #             locale_obj = babel.Locale(babel_locale)
    #             currencies = {
    #                 babel.numbers.get_currency_symbol(cn): cn
    #                 for cn in locale_obj.currency_symbols
    #             }

    #             self.currencies[babel_locale] = currencies

    #         currency_symbols = sorted(
    #             currencies, key=operator.length_hint, reverse=True
    #         )

    #         self.currency_symbols[babel_locale] = currency_symbols

    #     parsed = False
    #     for currency_symbol in currency_symbols:
    #         if word.text.startswith(currency_symbol):
    #             num_str = word.text[len(currency_symbol) :]
    #             try:
    #                 # Try to parse as a number
    #                 # This is important to handle thousand/decimal separators correctly.
    #                 number = babel.numbers.parse_decimal(num_str, locale=babel_locale)
    #                 word.interpret_as = InterpretAs.CURRENCY
    #                 word.currency_symbol = currency_symbol
    #                 word.number = number
    #                 parsed = True
    #                 break
    #             except ValueError:
    #                 pass

    #     if (not parsed) and (word.interpret_as == InterpretAs.CURRENCY):
    #         default_currency = default_currency or self.default_currency.get(
    #             babel_locale
    #         )
    #         if default_currency:
    #             # Forced interpretation using default currency
    #             try:
    #                 number = babel.numbers.parse_decimal(word.text, locale=babel_locale)
    #                 word.interpret_as = InterpretAs.CURRENCY
    #                 word.currency_name = default_currency
    #                 word.number = number
    #             except ValueError:
    #                 pass

    # def transform_initialism(self, node, graph):
    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if word.interpret_as:
    #         return

    #     if self.phonemizer and self.phonemizer.lookup(word.text):
    #         # Don't expand words already in lexicon
    #         return

    #     # TODO: Check alpha
    #     if (len(word.text) > 1) and word.text.isupper():
    #         word.interpret_as = InterpretAs.SPELL_OUT
    #     elif (len(word.text) > 3) and re.match(r"(?:[a-zA-Z]\.){2,}\s*", word.text):
    #         # TODO: Remove dots
    #         word.interpret_as = InterpretAs.SPELL_OUT

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
            num_str = re.sub(r"\W", " ", num_str).strip()

            # Split into separate words
            for part_str, sep_str in grouper(settings.split_pattern.split(num_str), 2):
                number_word_text = part_str
                if settings.keep_whitespace:
                    number_word_text += sep_str or ""

                if not number_word_text:
                    continue

                if settings.keep_whitespace and (not number_word_text.endswith(" ")):
                    number_word_text += " "

                number_word = WordNode(
                    node=len(graph),
                    implicit=True,
                    lang=word.lang,
                    text=number_word_text.strip(),
                    text_with_ws=number_word_text,
                )
                graph.add_node(number_word.node, data=number_word)
                graph.add_edge(word.node, number_word.node)

    # def verbalize_date(
    #     self, node, graph, babel_locale: typing.Optional[str] = None, **num2words_kwargs
    # ):
    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if (word.interpret_as != InterpretAs.DATE) or (word.date is None):
    #         return

    #     word_lang = word.lang or self.default_lang
    #     keep_whitespace = self.keep_whitespace.get(word_lang)
    #     if keep_whitespace is None:
    #         keep_whitespace = self.keep_whitespace.get(None)

    #     assert keep_whitespace is not None

    #     babel_locale = babel_locale or self.babel_locale
    #     if "lang" not in num2words_kwargs:
    #         num2words_kwargs["lang"] = self.num2words_lang

    #     date = word.date
    #     date_format = (word.format or InterpretAsFormat.DATE_MDY).strip().upper()
    #     day_str = ""
    #     month_str = ""
    #     year_str = ""

    #     if "M" in date_format:
    #         month_str = babel.dates.format_date(date, "MMMM", locale=babel_locale)

    #     if num2words_kwargs is None:
    #         num2words_kwargs = {}

    #     if "lang" not in num2words_kwargs:
    #         num2words_kwargs["lang"] = babel_locale

    #     if "D" in date_format:
    #         num2words_kwargs["to"] = "ordinal"
    #         day_str = num2words(date.day, **num2words_kwargs)

    #     if "Y" in date_format:
    #         num2words_kwargs["to"] = "year"
    #         year_str = num2words(date.year, **num2words_kwargs)

    #     # Transform into format string
    #     # MDY -> {M} {D} {Y}
    #     date_format_str = self.join_str.join(f"{{{c}}}" for c in date_format)
    #     date_str = date_format_str.format(M=month_str, D=day_str, Y=year_str)

    #     # Split into separate words
    #     for part_str, sep_str in grouper(self.split_pattern.split(date_str), 2):
    #         date_word_text = part_str
    #         if keep_whitespace:
    #             date_word_text += sep_str or ""

    #         if not date_word_text:
    #             continue

    #         if keep_whitespace and (not date_word_text.endswith(" ")):
    #             date_word_text += " "

    #         date_word = WordNode(
    #             node=len(graph),
    #             text=date_word_text.strip(),
    #             text_with_ws=date_word_text,
    #         )
    #         graph.add_node(date_word.node, data=date_word)
    #         graph.add_edge(word.node, date_word.node)

    # def verbalize_currency(
    #     self,
    #     node,
    #     graph,
    #     babel_locale: typing.Optional[str] = None,
    #     default_currency: typing.Optional[str] = None,
    #     currencies: typing.Optional[typing.Dict[str, str]] = None,
    #     **num2words_kwargs,
    # ):
    #     if not isinstance(node, WordNode):
    #         return

    #     word = typing.cast(WordNode, node)
    #     if (
    #         (word.interpret_as != InterpretAs.CURRENCY)
    #         or ((word.currency_symbol is None) and (word.currency_name is None))
    #         or (word.number is None)
    #     ):
    #         return

    #     babel_locale = babel_locale or word.lang or self.babel_locale
    #     default_currency = default_currency or self.default_currency.get(
    #         babel_locale, "USD"
    #     )
    #     decimal_num = word.number

    #     # True if number has non-zero fractional part
    #     num_has_frac = (decimal_num % 1) != 0

    #     if num2words_kwargs is None:
    #         num2words_kwargs = {}

    #     num2words_kwargs["to"] = "currency"

    #     if "lang" not in num2words_kwargs:
    #         num2words_kwargs["lang"] = babel_locale

    #     # Name of currency (e.g., USD)
    #     if not word.currency_name:
    #         currency_name = default_currency
    #         currencies = currencies or self.currencies.get(babel_locale)
    #         if currencies:
    #             # Look up currency in locale
    #             currency_name = currencies.get(word.currency_symbol, default_currency)

    #         word.currency_name = currency_name

    #     num2words_kwargs["currency"] = word.currency_name

    #     # Custom separator so we can remove 'zero cents'
    #     num2words_kwargs["separator"] = "|"

    #     num_str = num2words(float(decimal_num), **num2words_kwargs)

    #     # Post-process currency words
    #     if num_has_frac:
    #         # Discard num2words separator
    #         num_str = num_str.replace("|", "")
    #     else:
    #         # Remove 'zero cents' part
    #         num_str = num_str.split("|", maxsplit=1)[0]

    #     # Remove all non-word characters
    #     num_str = re.sub(r"\W", self.join_str, num_str).strip()

    #     # Split into separate words
    #     for part_str, sep_str in grouper(self.split_pattern.split(num_str), 2):
    #         currency_word_text = part_str
    #         if keep_whitespace:
    #             currency_word_text += sep_str or ""

    #         if not currency_word_text:
    #             continue

    #         if keep_whitespace and (not currency_word_text.endswith(" ")):
    #             currency_word_text += " "

    #         currency_word = WordNode(
    #             node=len(graph),
    #             text=currency_word_text.strip(),
    #             text_with_ws=currency_word_text,
    #         )
    #         graph.add_node(currency_word.node, data=currency_word)
    #         graph.add_edge(word.node, currency_word.node)

    @staticmethod
    def text_and_elements(element):
        yield element

        # Text before any tags (or end tag)
        text = element.text if element.text is not None else ""
        if text.strip():
            yield text

        for child in element:
            # Sub-elements
            yield from TextProcessor.text_and_elements(child)

        # End of current element
        yield TextProcessor.EndElement(element)

        # Text after the current tag
        tail = element.tail if element.tail is not None else ""
        if tail.strip():
            yield tail

    @staticmethod
    def print_graph(g, n, s: str = "-"):
        n_data = g.nodes[n]["data"]
        print(s, n, n_data, file=sys.stderr)
        for n2 in g.successors(n):
            TextProcessor.print_graph(g, n2, s + "-")


# -----------------------------------------------------------------------------


class TextProcessorTestCase(unittest.TestCase):
    def test_whitespace(self):
        processor = TextProcessor()
        graph, root = processor.process("This is  a   test    ")
        words = list(processor.words(graph, root))

        # Whitespace is retained by default
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="This", text_with_ws="This "),
                Word(idx=1, sent_idx=0, text="is", text_with_ws="is  "),
                Word(idx=2, sent_idx=0, text="a", text_with_ws="a   "),
                Word(idx=3, sent_idx=0, text="test", text_with_ws="test    "),
            ],
        )

    def test_no_whitespace(self):
        processor = TextProcessor(keep_whitespace=False)
        graph, root = processor.process("This is  a   test    ")
        words = list(processor.words(graph, root))

        # Whitespace is discarded
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="This", text_with_ws="This"),
                Word(idx=1, sent_idx=0, text="is", text_with_ws="is"),
                Word(idx=2, sent_idx=0, text="a", text_with_ws="a"),
                Word(idx=3, sent_idx=0, text="test", text_with_ws="test"),
            ],
        )

    def test_multiple_sentences(self):
        processor = TextProcessor(major_breaks={".", "!"})
        graph, root = processor.process("First sentence. Second sentence!")
        words = list(processor.words(graph, root))

        # Separated by a major break
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="First", text_with_ws="First "),
                Word(idx=1, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(idx=0, sent_idx=1, text="Second", text_with_ws="Second "),
                Word(idx=1, sent_idx=1, text="sentence", text_with_ws="sentence"),
            ],
        )

    def test_explicit_sentence(self):
        processor = TextProcessor(major_breaks={".", "!"})
        graph, root = processor.process("<s>First sentence. Second sentence!</s>")
        words = list(processor.words(graph, root))

        # Sentences should not be split apart
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="First", text_with_ws="First "),
                Word(idx=1, sent_idx=0, text="sentence", text_with_ws="sentence"),
                Word(idx=2, sent_idx=0, text="Second", text_with_ws="Second "),
                Word(idx=3, sent_idx=0, text="sentence", text_with_ws="sentence"),
            ],
        )

    def test_numbers_one_language(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor.process("1 2 3")
        words = list(processor.words(graph, root))

        # Numbers should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(idx=1, sent_idx=0, text="two", text_with_ws="two "),
                Word(idx=2, sent_idx=0, text="three", text_with_ws="three "),
            ],
        )

    def test_numbers_multiple_languages(self):
        processor = TextProcessor(default_lang="en_US")
        graph, root = processor.process('1 <w lang="es_ES">2</w> <w lang="de_DE">3</w>')
        words = list(processor.words(graph, root))

        # Numbers should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="one", text_with_ws="one "),
                Word(idx=1, sent_idx=0, text="dos", text_with_ws="dos "),
                Word(idx=2, sent_idx=0, text="drei", text_with_ws="drei "),
            ],
        )

    def test_currency_one_language(self):
        processor = TextProcessor(lang="en_US")
        graph, root = processor.process("$10")
        words = list(processor.words(graph, root))

        # Currency should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(idx=1, sent_idx=0, text="dollars", text_with_ws="dollars "),
            ],
        )

    def test_currency_multiple_language(self):
        processor = TextProcessor(lang="en_US")
        graph, root = processor.process(
            '€10 <w lang="fr_FR">€10</w> <w lang="nl_NL">€10</w>'
        )
        words = list(processor.words(graph, root))

        # Currencies should be verbalized
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(idx=1, sent_idx=0, text="euro", text_with_ws="euro "),
                Word(idx=2, sent_idx=0, text="dix", text_with_ws="dix "),
                Word(idx=3, sent_idx=0, text="euros", text_with_ws="euros "),
                Word(idx=4, sent_idx=0, text="tien", text_with_ws="tien "),
                Word(idx=5, sent_idx=0, text="euro", text_with_ws="euro "),
            ],
        )

    def test_currency_default(self):
        processor = TextProcessor(lang="en_US", default_currency={"en_US": "USD"})
        graph, root = processor.process('<say-as interpret-as="currency">10</say-as>')
        words = list(processor.words(graph, root))

        # Currency should be verbalized, despite lack of "$" symbol
        self.assertEqual(
            words,
            [
                Word(idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
                Word(idx=1, sent_idx=0, text="dollars", text_with_ws="dollars "),
            ],
        )

    def test_phonemize_one_language(self):
        processor = TextProcessor(
            lang="en_US",
            phonemizer=Phonemizer(
                lang="en_US",
                lexicon={
                    "en_US": {"test": {Phonemizer.DEFAULT_ROLE: ["t", "e", "s", "t"]}}
                },
            ),
        )
        graph, root = processor.process("test")
        words = list(processor.words(graph, root))

        # Single word is phonemized according to the lexicon
        self.assertEqual(
            words,
            [
                Word(
                    idx=0,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["t", "e", "s", "t"],
                ),
            ],
        )

    def test_phonemize_one_language_multiple_roles(self):
        processor = TextProcessor(
            lang="en_US",
            phonemizer=Phonemizer(
                lang="en_US",
                lexicon={
                    "en_US": {
                        "test": {
                            # Made-up roles.
                            # These are usually part-of-speech tags.
                            "role_1": ["t", "e", "s", "t"],
                            "role_2": ["T", "E", "S", "T"],
                        }
                    }
                },
            ),
        )

        # Use both made-up roles
        graph, root = processor.process(
            '<speak><w role="role_1">test</w> <w role="role_2">test</w></speak>'
        )
        words = list(processor.words(graph, root))

        # Single word is phonemized according to the lexicon in two different manners
        self.assertEqual(
            words,
            [
                Word(
                    idx=0,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["t", "e", "s", "t"],
                ),
                Word(
                    idx=1,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["T", "E", "S", "T"],
                ),
            ],
        )

    def test_phonemize_multiple_languages(self):
        processor = TextProcessor(
            lang="en_US",
            phonemizer=Phonemizer(
                lang="en_US",
                lexicon={
                    # Made-up phonemes for two different languages
                    "en_US": {"test": {Phonemizer.DEFAULT_ROLE: ["t", "e", "s", "t"]}},
                    "de_DE": {"test": {Phonemizer.DEFAULT_ROLE: ["T", "E", "S", "T"]}},
                },
            ),
        )
        graph, root = processor.process(
            '<speak><w lang="en_US">test</w> <w lang="de_DE">test</w></speak>'
        )
        words = list(processor.words(graph, root))

        # Single word is phonemized according to the lexicon with two different languages
        self.assertEqual(
            words,
            [
                Word(
                    idx=0,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["t", "e", "s", "t"],
                ),
                Word(
                    idx=1,
                    sent_idx=0,
                    text="test",
                    text_with_ws="test",
                    phonemes=["T", "E", "S", "T"],
                ),
            ],
        )

    # def test_pos_phonemize(self):
    #     processor = TextProcessor(lang="en_US", default_currency={"en_US": "USD"})
    #     graph, root = processor.process('<say-as interpret-as="currency">10</say-as>')
    #     words = list(processor.words(graph, root))

    #     # Currency should be verbalized, despite lack of "$" symbol
    #     self.assertEqual(
    #         words,
    #         [
    #             Word(idx=0, sent_idx=0, text="ten", text_with_ws="ten "),
    #             Word(idx=1, sent_idx=0, text="dollars", text_with_ws="dollars "),
    #         ],
    #     )

    # def test_no_whitespace(self):
    #     processor = TextProcessor(keep_whitespace=False)
    #     graph, root = processor.process("This is  a   test    ")
    #     words = list(processor.words(graph, root))

    #     # Whitespace is discarded
    #     self.assertEqual(
    #         words,
    #         [
    #             Word(text="This", text_with_ws="This"),
    #             Word(text="is", text_with_ws="is"),
    #             Word(text="a", text_with_ws="a"),
    #             Word(text="test", text_with_ws="test"),
    #         ],
    #     )

    # def test1(self):
    #     tokenizer = TextProcessor()
    #     tokenizer.tokenize(
    #         "<speak>This is a test. <p><s>These are.</s><s>Two sentences.</s></p></speak>"
    #     )

    # def test2(self):
    #     tokenizer = TextProcessor()
    #     tokenizer.tokenize("<speak>1/2/2022 1,234 $50.32</speak>")

    # def test3(self):
    #     tokenizer = TextProcessor()
    #     tokenizer.tokenize(
    #         '<speak><say-as interpret-as="number" format="ordinal">12</say-as></speak>'
    #     )

    # def test4(self):
    #     tokenizer = TextProcessor(
    #         pos_model="/home/hansenm/opt/gruut/gruut/data/en-us/pos/model.crf"
    #     )
    #     tokenizer.tokenize("This is a <break /> $50 test")

    # def test5(self):

    #     tokenizer = TextProcessor(
    #         major_breaks={"."},
    #         pos_model="/home/hansenm/opt/gruut/gruut/data/en-us/pos/model.crf",
    #         phonemizer=Phonemizer(
    #             db_conn=sqlite3.connect(
    #                 "/home/hansenm/opt/gruut/gruut/data/en-us/lexicon.db"
    #             ),
    #             g2p_model="/home/hansenm/opt/gruut/gruut/data/en-us/g2p/model.crf",
    #             word_transform_funcs=[str.lower],
    #             guess_transform_func=str.lower,
    #         ),
    #     )

    #     tokenizer.tokenize("<speak>plOOP</speak>")
