"""Class for cleaning raw text and tokenizing"""
import logging
import re
import typing
from dataclasses import dataclass, field

import babel.numbers
import pydash
import spacy
from num2words import num2words
from spacy.tokens import Doc

_LOGGER = logging.getLogger("gruut.toksen")

# -----------------------------------------------------------------------------


@dataclass
class Sentence:
    """Tokenized and cleaned sentence"""

    raw_text: str = ""
    raw_words: typing.List[str] = field(default_factory=list)
    clean_words: typing.List[str] = field(default_factory=list)


class Tokenizer:
    """Splits text into sentences, tokenizes and cleans"""

    def __init__(self, config, nlp=None):
        self.config = config
        self.language = pydash.get(self.config, "language.code")

        # Short pause symbols (commas, etc.)
        self.minor_breaks: typing.Set[str] = set(
            pydash.get(self.config, "symbols.minor_breaks", [])
        )

        # End of sentence symbol
        self.major_break: typing.Optional[str] = pydash.get(
            self.config, "symbols.major_break"
        )

        # If True, keep question marks
        self.question_mark = bool(
            pydash.get(self.config, "symbols.question_mark", False)
        )

        # Regex to match numbers (digits)
        self.number_pattern: typing.Optional[re.Pattern] = None
        number_regex = pydash.get(self.config, "symbols.number_regex")
        if number_regex:
            self.number_pattern = re.compile(number_regex)

        self.babel_locale = pydash.get(
            self.config, "symbols.babel_locale", self.language.replace("-", "_")
        )
        self.num2words_lang = pydash.get(
            self.config, "symbols.num2words_lang", self.language
        )

        # Case transformation (lower/upper)
        casing = pydash.get(self.config, "symbols.casing")
        self.casing: typing.Optional[typing.Callable[[str], str]] = None
        if casing == "lower":
            self.casing = str.lower
        elif casing == "upper":
            self.casing = str.upper

        # [(pattern, replacement)]
        self.replacements = []
        for replace_key, replace_value in pydash.get(
            self.config, "symbols.replace", {}
        ).items():
            self.replacements.append((re.compile(replace_key), replace_value))

        # short form -> [expansion words]
        self.abbreviations: typing.Dict[str, typing.List[str]] = {}
        for abbrev_key, abbrev_value in self.config.get("abbreviations", {}).items():
            if isinstance(abbrev_value, str):
                # One word expansion
                abbrev_value = [abbrev_value]

            # short form -> [expansion words]
            self.abbreviations[abbrev_key] = abbrev_value

        if nlp:
            self.nlp = nlp
        else:
            # Load spaCy model
            _LOGGER.debug("Loading spaCy model for %s", self.language)
            self.nlp = spacy.load(self.language, disable=["tagger", "parser", "ner"])
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))

            self.tokenizer_exclude = set(
                pydash.get(self.config, "symbols.tokenizer_exclude", [])
            )

            if self.tokenizer_exclude:
                # Exclude tokenizer rules with specific characters (e.g., apostrophes for contractions)
                # https://stackoverflow.com/questions/59579049/how-to-tell-spacy-not-to-split-any-words-with-apostrophs-using-retokenizer
                self.nlp.tokenizer.rules = {
                    key: value
                    for key, value in self.nlp.tokenizer.rules.items()
                    if self.tokenizer_exclude.isdisjoint(set(key))
                }

    def tokenize(self, text: str) -> typing.Iterable[Sentence]:
        """Split text into sentences, tokenize, and clean"""
        # Do pre-tokenization replacements
        for pattern, replacement in self.replacements:
            text = pattern.sub(replacement, text)

        # Increase max length as necessary.
        # This should be fine since we're not using the parser or NER.
        self.nlp.max_length = max(self.nlp.max_length, len(text))

        doc: Doc = self.nlp(text)

        # Process each sentence
        for sentence in doc.sents:
            raw_words = []
            clean_words = []

            # Process each token
            for token in sentence:
                raw_words.append(token.text)

                if (token.text in self.minor_breaks) or (
                    token.text == self.major_break
                ):
                    # Keep breaks (pauses)
                    clean_words.append(token.text)
                    continue

                if self.question_mark and token.text == "?":
                    # Keep question marks
                    clean_words.append(token.text)
                    continue

                if (
                    token.is_space
                    or token.is_punct
                    or token.like_url
                    or token.like_email
                ):
                    # Skip non-words
                    continue

                process_as_word = True

                # Try to process as a number first
                if self.number_pattern and self.number_pattern.match(token.text):
                    try:
                        num = babel.numbers.parse_decimal(
                            token.text, locale=self.babel_locale
                        )
                        num_str = num2words(num, lang=self.num2words_lang)

                        # Tokenize number string itself
                        num_doc = self.nlp(num_str)
                        for num_sent in num_doc.sents:
                            for num_token in num_sent:
                                if num_token.is_space or num_token.is_punct:
                                    # Skip non-words
                                    continue

                                clean_words.append(num_token.text)

                        # Successfully processed as a number
                        process_as_word = False
                    except Exception:
                        _LOGGER.exception(token.text)

                if process_as_word:
                    # Not a number
                    words = [token.text]

                    # Apply casing transformation
                    if self.casing:
                        words = [self.casing(w) for w in words]

                    # Expand abbreviations
                    if self.abbreviations:
                        expanded_words = []
                        for word in words:
                            # Expansions may be multiple words.
                            # They will not having casing/replacements applied.
                            expansion = self.abbreviations.get(word)
                            if expansion:
                                expanded_words.extend(expansion)
                            else:
                                expanded_words.append(word)

                        words = expanded_words

                    clean_words.extend(words)

            yield Sentence(
                raw_text=sentence.text, raw_words=raw_words, clean_words=clean_words
            )
