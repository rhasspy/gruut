#!/usr/bin/env python3
from dataclasses import dataclass
import functools
import typing
import re
import sys
import itertools
import unittest
from enum import Enum
from collections import defaultdict
from decimal import Decimal
from datetime import datetime

import dateparser
import babel
import babel.numbers
import babel.dates
import networkx as nx
from num2words import num2words


from .const import REGEX_MATCH, REGEX_PATTERN, REGEX_TYPE, Token, Sentence
from .toksen import Tokenizer
from .utils import grouper, sliding_window, get_currency_names, maybe_compile_regex

# -----------------------------------------------------------------------------

DEFAULT_MAJOR_BREAKS = set(".?!")
DEFAULT_MINOR_BREAKS = set(",;:")
DEFAULT_SPLIT_PATTERN = re.compile("(\s+)")
DEFAULT_NON_WORD_PATTERN = re.compile(r"^(\W|_)+$")


@dataclass
class Span:
    left: int
    right: int
    text: str
    depth: int = 0


@dataclass
class MajorBreakSpan(Span):
    pass


@dataclass
class MinorBreakSpan(Span):
    pass


@dataclass
class CurrencySpan(Span):
    currency_name: typing.Optional[str] = None


@dataclass
class NumberSpan(Span):
    number: Decimal = Decimal()


@dataclass
class DateSpan(Span):
    date: datetime = datetime.now()


# -----------------------------------------------------------------------------


class GraphTokenizer(Tokenizer):
    def __init__(
        self,
        locale="en_US",
        babel_locale: typing.Optional[str] = None,
        num2words_lang: typing.Optional[str] = None,
        dateparser_lang: typing.Optional[str] = None,
        major_breaks: typing.Optional[typing.Set[str]] = None,
        minor_breaks: typing.Optional[typing.Set[str]] = None,
        split_pattern: REGEX_TYPE = DEFAULT_SPLIT_PATTERN,
        non_word_pattern: typing.Optional[REGEX_TYPE] = DEFAULT_NON_WORD_PATTERN,
        abbreviations: typing.Optional[typing.Dict[str, typing.List[str]]] = None,
        currencies: typing.Optional[typing.Dict[str, str]] = None,
        replace_numbers: bool = True,
        replace_currency: bool = True,
        replace_dates: bool = True,
        join_str: str = " ",
        replacements: typing.Optional[
            typing.Sequence[typing.Tuple[REGEX_TYPE, str]]
        ] = None,
        punctuations: typing.Optional[typing.Set[str]] = None,
    ):

        if babel_locale is None:
            babel_locale = locale

        if num2words_lang is None:
            num2words_lang = locale

        if dateparser_lang is None:
            # en_US -> en
            dateparser_lang = locale.split("_")[0]

        self.locale = locale
        self.babel_locale = babel_locale
        self.num2words_lang = num2words_lang
        self.dateparser_lang = dateparser_lang

        self.punctuations = punctuations or set()
        self.major_breaks = major_breaks or set()
        self.minor_breaks = minor_breaks or set()

        breaks = set.union(self.major_breaks, self.minor_breaks)
        breaks_str = "|".join(re.escape(p) for p in breaks)

        self.abbreviation_patterns = {}
        if abbreviations:
            for text_or_pattern, replacement_words in abbreviations.items():
                if isinstance(text_or_pattern, str):
                    text = text_or_pattern
                    if not text.startswith("^"):
                        text = f"^{text}"

                    if not text.endswith("$"):
                        # Automatically add ending punctuation and whitespace
                        text = f"{text}(?:{breaks_str})?\\s*$"

                    pattern = re.compile(text)
                    self.abbreviation_patterns[pattern] = replacement_words
                else:
                    self.abbreviation_patterns[text_or_pattern] = replacement_words

        self.split_pattern = maybe_compile_regex(split_pattern)
        self.replacements = [
            (maybe_compile_regex(p), r) for p, r in (replacements or [])
        ]
        self.non_word_pattern = (
            maybe_compile_regex(non_word_pattern) if non_word_pattern else None
        )
        self.replace_numbers = replace_numbers
        self.replace_currency = replace_currency
        self.replace_dates = replace_dates

        if (not currencies) and replace_currency:
            currencies = get_currency_names(babel_locale)

        self.currencies = currencies

        self.join_str = join_str

    def pre_tokenize(self, text: str) -> str:
        for pattern, replacement in self.replacements:
            text = pattern.sub(replacement, text)

        return text

    def tokenize(self, text: str, **kwargs) -> typing.Iterable[Sentence]:
        text = self.pre_tokenize(text)
        g, root = self.text_to_graph(text)
        word_nodes = list(g.successors(root))

        raw_word_nodes = []
        tokens = []

        for n in itertools.chain(leaves(g, root, skip_seen=True), [None]):
            sentence_ready = False

            if n is None:
                sentence_ready = True

                # Consume remaining word nodes
                raw_word_nodes.extend(word_nodes)
                word_nodes = []
            else:
                n_span = g.nodes[n]["span"]

                token = Token(text=n_span.text)
                tokens.append(token)

                if isinstance(n_span, MajorBreakSpan):
                    sentence_ready = True

                # Accumulate raw words
                raw_node = n
                while not g.has_edge(root, raw_node):
                    raw_node = next(iter(g.predecessors(raw_node)))

                while word_nodes and (word_nodes[0] != raw_node):
                    raw_word_nodes.append(word_nodes[0])
                    word_nodes = word_nodes[1:]

            if sentence_ready and tokens:
                # Do post-processing
                tokens = self.post_tokenize(tokens)

                # Text as it was before pipeline
                raw_words = [g.nodes[raw_n]["span"].text for raw_n in raw_word_nodes]
                raw_text = "".join(raw_words)

                # Text after pipeline and only if is_word is True
                assert False, tokens
                clean_words = [t.text for t in tokens if self.is_word(t.text)]
                clean_text = self.join_str.join(clean_words)

                yield Sentence(
                    tokens=tokens,
                    clean_words=clean_words,
                    clean_text=clean_text,
                    raw_words=raw_words,
                    raw_text=raw_text,
                )

                # Reset state for next sentence
                tokens = []
                raw_word_nodes = []

    def text_to_graph(self, text: str, **kwargs) -> typing.Tuple[nx.DiGraph, int]:
        g = nx.DiGraph()
        root = 0
        g.add_node(root, span=Span(left=0, right=len(text), text=text))

        # Break apart at whitespace
        pipeline_tokenize(g, root, self.split_pattern)

        if self.abbreviation_patterns:
            # Expand abbreviations into multiple words
            pipeline_transform(
                g,
                root,
                functools.partial(
                    transform_abbreviations, abbreviations=self.abbreviation_patterns
                ),
            )

        if self.punctuations:
            punctuations_str = "|".join(re.escape(p) for p in self.punctuations)
            pipeline_split(
                g, root, re.compile(f"({punctuations_str})"),
            )

            print_graph(g, root)

        if self.currencies and self.replace_currency:
            currency_symbols = "|".join(re.escape(c) for c in self.currencies)
            pipeline_split(
                g, root, re.compile(f"^({currency_symbols})"), classes=[CurrencySpan],
            )

        if self.major_breaks:
            pipeline_split(
                g, root, re.compile(r"([.?!])\s*$"), classes=[Span, MajorBreakSpan]
            )

        if self.minor_breaks:
            pipeline_split(
                g, root, re.compile(r"([,:;])\s*$"), classes=[Span, MinorBreakSpan]
            )

        if self.replace_numbers:
            pipeline_transform(g, root, transform_number)

        if self.replace_dates:
            pipeline_transform(
                g,
                root,
                functools.partial(
                    transform_date, settings={"STRICT_PARSING": True}, languages=["en"],
                ),
            )

        if self.replace_currency:
            pipeline_transform_window(
                g,
                root,
                2,
                functools.partial(verbalize_currency_number, lang=self.num2words_lang),
            )

        if self.replace_numbers:
            pipeline_transform(
                g, root, functools.partial(verbalize_number, lang=self.num2words_lang)
            )

        if self.replace_dates:
            pipeline_transform(
                g,
                root,
                functools.partial(verbalize_date, babel_locale=self.babel_locale),
            )

        return g, root

    def is_word(self, text: str) -> bool:
        """
        Determine if text is a word or not.

        Args:
            text: Text to check

        Returns:
            `True` if text is considered a word
        """
        text = text.strip()
        if not text:
            # Empty string
            return False

        if self.non_word_pattern:
            word_match = self.non_word_pattern.match(text)

            if word_match is not None:
                # Matches non-word regex
                return False

        return (
            len(text) > 0
            and (text not in self.minor_breaks)
            and (text not in self.major_breaks)
            and (text not in self.punctuations)
        )


# -----------------------------------------------------------------------------


def print_graph(g, n, s="-"):
    n_span = g.nodes[n]["span"]
    print(s, n, n_span, file=sys.stderr)
    for n2 in g.successors(n):
        print_graph(g, n2, s + "-")


def leaves(g, n, skip_seen=False, seen=None):
    if skip_seen and (seen is None):
        seen = set()

    if g.out_degree(n) == 0:
        if not skip_seen or (n not in seen):
            yield n

            if seen is not None:
                seen.add(n)
    else:
        for m in g.successors(n):
            yield from leaves(g, m)


def pipeline_split(g, root, pattern, classes=None):
    for n in list(leaves(g, root)):
        n_span = g.nodes[n]["span"]
        parts = pattern.split(n_span.text)
        if len(parts) < 2:
            continue

        offset = 0
        class_idx = 0
        for part in parts:
            if not part:
                continue

            span_class = Span
            if classes and (class_idx < len(classes)):
                span_class = classes[class_idx]

            next_node = len(g)
            g.add_node(
                next_node,
                span=span_class(
                    left=offset,
                    right=offset + len(part),
                    text=part,
                    depth=n_span.depth + 1,
                ),
            )
            g.add_edge(n, next_node)

            offset += len(part)
            class_idx += 1


def transform_abbreviations(n_span, abbreviations):
    for p, r in abbreviations.items():
        match = re.match(p, n_span.text)
        if match is not None:
            offset = 0
            for new_word in r:
                new_word += " "
                yield Span(
                    left=offset,
                    right=offset + len(new_word),
                    text=new_word,
                    depth=n_span.depth + 1,
                )

                offset += len(new_word)

            start, end = match.start(1), match.end(1)
            if start > 0:
                # before
                yield Span(
                    left=0,
                    right=start,
                    text=n_span.text[: start + 1],
                    depth=n_span.depth + 1,
                )

            if end < (len(n_span.text) - 1):
                # after
                yield Span(
                    left=0, right=start, text=n_span.text[end:], depth=n_span.depth + 1,
                )


def transform_number(n_span, babel_locale="en_US"):
    try:
        # Try to parse as a number
        # This is important to handle thousand/decimal separators correctly.
        number = babel.numbers.parse_decimal(n_span.text, locale=babel_locale)
        yield NumberSpan(
            left=0,
            right=len(n_span.text),
            depth=n_span.depth + 1,
            text=n_span.text,
            number=number,
        )
    except ValueError:
        pass


def transform_date(n_span, **dateparser_kwargs):
    date = dateparser.parse(n_span.text, **dateparser_kwargs)
    if date is not None:
        yield DateSpan(
            left=0,
            right=len(n_span.text),
            depth=n_span.depth + 1,
            text=n_span.text,
            date=date,
        )


def pipeline_tokenize(g, root, pattern):
    for n in list(leaves(g, root)):
        n_span = g.nodes[n]["span"]

        last_index = 0
        for non_ws, ws in grouper(pattern.split(n_span.text), 2):
            word = non_ws + (ws or "")
            if not word:
                continue

            next_index = last_index + len(word)
            next_node = len(g)
            g.add_node(
                next_node,
                span=Span(left=last_index, right=next_index, text=word, depth=1),
            )
            g.add_edge(root, next_node)

            last_index = next_index


def pipeline_transform(g, root, transform):
    for n in list(leaves(g, root)):
        n_span = g.nodes[n]["span"]
        for new_span in transform(n_span):
            next_node = len(g)
            g.add_node(next_node, span=new_span)
            g.add_edge(n, next_node)


def pipeline_transform_window(g, root, window_size, transform):
    for n_window in sliding_window(list(leaves(g, root)), n=window_size):
        n_spans = [g.nodes[n]["span"] for n in n_window if n is not None]
        for new_span in transform(n_spans):
            next_node = len(g)
            g.add_node(next_node, span=new_span)

            for n in n_window:
                g.add_edge(n, next_node)


def verbalize_number(n_span, **num2words_kwargs):
    if isinstance(n_span, NumberSpan):
        decimal_num = n_span.number
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
        last_index = 0
        for a, b in grouper(re.split(r"(\s+)", num_str), 2):
            word = a + (b or "")
            if not word:
                continue

            if not word.endswith(" "):
                word += " "

            next_index = last_index + len(word)

            yield Span(
                left=last_index, right=next_index, text=word, depth=n_span.depth + 1
            )
            last_index = next_index


def verbalize_date(n_span, babel_locale="en_US", **num2words_kwargs):
    if isinstance(n_span, DateSpan):
        date = n_span.date
        month_str = babel.dates.format_date(date, "MMMM", locale=babel_locale)

        if num2words_kwargs is None:
            num2words_kwargs = {}

        if "lang" not in num2words_kwargs:
            num2words_kwargs["lang"] = babel_locale

        num2words_kwargs["to"] = "ordinal"
        day_str = num2words(date.day, **num2words_kwargs)

        num2words_kwargs["to"] = "year"
        year_str = num2words(date.year, **num2words_kwargs)

        date_str = f"{month_str} {day_str} {year_str}"

        # Split into separate words
        last_index = 0
        for a, b in grouper(re.split(r"(\s+)", date_str), 2):
            word = a + (b or "")
            if not word:
                continue

            if not word.endswith(" "):
                word += " "

            next_index = last_index + len(word)

            yield Span(left=last_index, right=next_index, text=word, depth=n_span.depth)
            last_index = next_index


def verbalize_currency_number(
    n_spans,
    default_currency: str = "USD",
    currency_names: typing.Optional[typing.Dict[str, str]] = None,
    **num2words_kwargs,
):
    if (
        (len(n_spans) == 2)
        and isinstance(n_spans[0], CurrencySpan)
        and isinstance(n_spans[1], NumberSpan)
    ):
        currency_span = typing.cast(CurrencySpan, n_spans[0])
        number_span = typing.cast(NumberSpan, n_spans[1])
        new_depth = max(currency_span.depth, number_span.depth) + 1

        decimal_num = number_span.number

        # True if number has non-zero fractional part
        num_has_frac = (decimal_num % 1) != 0

        if num2words_kwargs is None:
            num2words_kwargs = {}

        num2words_kwargs["to"] = "currency"

        # Name of currency (e.g., USD)
        currency_name = default_currency
        if currency_names:
            currency_name = currency_names.get(currency_span.text, default_currency)

        num2words_kwargs["currency"] = currency_name

        # Custom separator so we can remove 'zero cents'
        num2words_kwargs["separator"] = "|"

        num_str = num2words(float(decimal_num), **num2words_kwargs)

        # Post-process currency words
        if num_has_frac:
            # Discard num2words separator
            num_str = num_str.replace("|", "")
        else:
            # Remove 'zero cents' part
            num_str = num_str.split("|", maxsplit=1)[0]

        # Remove all non-word characters
        num_str = re.sub(r"\W", " ", num_str).strip()

        # Split into separate words
        last_index = 0
        for a, b in grouper(re.split(r"(\s+)", num_str), 2):
            word = a + (b or "")
            if not word:
                continue

            if not word.endswith(" "):
                word += " "

            next_index = last_index + len(word)

            yield Span(left=last_index, right=next_index, text=word, depth=new_depth)
            last_index = next_index
