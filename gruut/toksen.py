"""Class for cleaning raw text and tokenizing"""
import abc
import logging
import re
import typing
from decimal import Decimal
from pathlib import Path

from num2words import num2words

from .const import (
    REGEX_MATCH,
    REGEX_PATTERN,
    REGEX_TYPE,
    Sentence,
    Token,
    TokenFeatures,
)
from .pos import PartOfSpeechTagger
from .utils import maybe_compile_regex

_LOGGER = logging.getLogger("gruut.toksen")

# -----------------------------------------------------------------------------


class Tokenizer(abc.ABC):
    """Abstract base class for tokenizers"""

    # pylint: disable=R0201
    def pre_tokenize(self, text: str) -> str:
        """
        Pre-process text before tokenization (called in :py:meth:`tokenize`).

        Args:
            text: Text to pre-process

        Returns:
            Pre-processed text
        """
        return text

    @abc.abstractmethod
    def tokenize(self, text: str, **kwargs) -> typing.Iterable[Sentence]:
        """
        Split text into tokenized sentences (with pre/post processing).

        Args:
            text: Text to tokenize

        Returns:
            sentences
        """
        pass

    # pylint: disable=R0201
    def post_tokenize(self, tokens: typing.Sequence[Token]) -> typing.Sequence[Token]:
        """
        Post-process tokens (called in :py:meth:`tokenize`)

        Args:
            tokens: Tokens to post-process

        Returns:
            Post-processed tokens
        """
        return tokens


# -----------------------------------------------------------------------------


class RegexTokenizer(Tokenizer):
    """
    Full-featured tokenizer with support for number expansion.

    Pipline is (roughly):

    #. pre_tokenize applies regex replacements to raw text
    #. text is split using split_pattern into words

       * abbreviations are expanded, text is re-split

    #. words are split into sub-words using punctuations

       * sub-words are grouped by sentence and converted to Tokens

    #. Tokens are cleaned

       * empty and non-word tokens are dropped
       * numbers are expanded to words
       * casing_func is applied

    #. Part of speech tags are predicted (if model available)


    Attributes:
        split_pattern: `str` or :py:class:`re.Pattern` used to split text into words
        join_str: `str` used to join words back into text
        replacements: (:py:class:`re.Pattern`, replacement `str`) tuples used in pre_tokenize on text
        casing_func: Function applied during token cleaning and abbreviation expansion (e.g., :py:meth:`str.lower`).
        punctuations: Single-character strings that cause words to split into sub-words. Punctuation tokens are dropped from sentence tokens unless its a major/minor break.
        minor_breaks: Single-character strings that indicate short pauses in a sentence. Minor break tokens are kept in sentence tokens.
        major_breaks: Single-character strings that indicate the start of a new sentence. Major break tokens are kept in sentence tokens.
        abbreviations: Short/long form mapping. Text is resplit on whitespace after expansion. If key is a `str`, optional punctuation is automatically added around the pattern. If key is an :py:class:`re.Pattern`, it must be suitable for use with :py:meth:`re.subn` with `count=1`.
        number_pattern: Pattern used to match numbers. First group must capture the number text.
        number_converter_pattern: Pattern used to match numbers with a converter specified. For example, "2021_year" will expanded to "twenty twenty one" instead of "two thousand twenty one". First group must capture the number text, second group the converter.
        non_word_pattern: Pattern used to match non-words, which will be excluded from sentence tokens if `exclude_non_words=True`
        exclude_non_words: Excludes non words from sentence tokens (see :py:meth:`RegexTokenizer.is_word`)
        num2words_lang: Language for num2words number expansion (e.g., "en_US")
        babel_locale: Locale for babel number parsing (e.g., "en_US")
        use_number_converters: If `True`, numbers may contain converters (see `number_converter_pattern`).
        currency_names: Mapping from currency symbol (e.g., "$") to currency name (e.g., "USD"). Used by `num2words` during number expansion. Currency symbols are also treated as punctuation during sub-word tokenization.
        do_replace_currency: If `True`, numbers after a currency symbol are converted using the `num2words` "currency" converter. See `currency_names`.
        currency_replacements: (:py:class:`re.Pattern`, replacement) tuples that are applied to the string returned by `num2words` after currency conversion. Example: "$1.50" becomes "one dollar, fifty cents", so you may want to replace "," with " and ".
        pos_model: Path to CRF part of speech tagger model. See also: :py:mod:`gruut.pos`.
    """

    WHITESPACE_PATTERN = re.compile(r"\s+")
    """Default pattern for initially splitting text into words"""

    NON_WORD_PATTERN = re.compile(r"^(\W|_)+$")
    """Default pattern for matching non-words"""

    NUMBER_PATTERN = re.compile(r"^(-?\d+(?:[,.]\d+)*)$")
    """
    Default pattern for matching numbers like 3.14.
    First group must be number text ("3.14").
    """

    NUMBER_CONVERTER_PATTERN = re.compile(r"^(-?\d+(?:[,.]\d+)*)_(\w+)$")
    """
    Default pattern for matching numbers with a "converter" like "1970_year".
    First group must be number text ("1970").
    Second group must be converter ("year").
    """

    def __init__(
        self,
        split_pattern: REGEX_TYPE = WHITESPACE_PATTERN,
        join_str: str = " ",
        replacements: typing.Optional[
            typing.Sequence[typing.Tuple[REGEX_TYPE, str]]
        ] = None,
        casing_func: typing.Optional[typing.Callable[[str], str]] = None,
        punctuations: typing.Optional[typing.Set[str]] = None,
        minor_breaks: typing.Optional[typing.Set[str]] = None,
        major_breaks: typing.Optional[typing.Set[str]] = None,
        abbreviations: typing.Optional[
            typing.Mapping[typing.Union[str, REGEX_PATTERN], str]
        ] = None,
        number_pattern: REGEX_TYPE = NUMBER_PATTERN,
        number_converter_pattern: REGEX_TYPE = NUMBER_CONVERTER_PATTERN,
        non_word_pattern: typing.Optional[REGEX_TYPE] = NON_WORD_PATTERN,
        exclude_non_words: bool = True,
        num2words_lang: typing.Optional[str] = None,
        babel_locale: typing.Optional[str] = None,
        use_number_converters: bool = False,
        currency_names: typing.Optional[typing.Dict[str, str]] = None,
        do_replace_currency: bool = False,
        currency_replacements: typing.Optional[
            typing.Sequence[typing.Tuple[REGEX_TYPE, str]]
        ] = None,
        pos_model: typing.Optional[typing.Union[str, Path]] = None,
    ):
        self.split_pattern = maybe_compile_regex(split_pattern)
        self.join_str = join_str
        self.replacements = [
            (maybe_compile_regex(p), r) for p, r in (replacements or [])
        ]
        self.casing_func = casing_func
        self.punctuations = punctuations or set()
        self.minor_breaks = minor_breaks or set()
        self.major_breaks = major_breaks or set()
        self.number_pattern = maybe_compile_regex(number_pattern)
        self.number_converter_pattern = (
            maybe_compile_regex(number_converter_pattern)
            if number_converter_pattern
            else None
        )
        self.non_word_pattern = (
            maybe_compile_regex(non_word_pattern) if non_word_pattern else None
        )
        self.exclude_non_words = exclude_non_words
        self.num2words_lang = num2words_lang
        self.babel_locale = babel_locale
        self.use_number_converters = use_number_converters
        self.currency_names = currency_names or {}
        self.do_replace_currency = do_replace_currency
        self.currency_replacements = [
            (maybe_compile_regex(p), r) for p, r in (currency_replacements or [])
        ]

        self.pos_tagger: typing.Optional[PartOfSpeechTagger] = None
        if pos_model is not None:
            self.pos_tagger = PartOfSpeechTagger(pos_model)

        # Must be after self.punctuations is set
        self.abbreviations = self._make_abbreviation_patterns(abbreviations or {})

    def pre_tokenize(self, text: str) -> str:
        for pattern, replacement in self.replacements:
            text = pattern.sub(replacement, text)

        return text

    def tokenize(self, text: str, **kwargs) -> typing.Iterable[Sentence]:
        # Pre-processing
        text = self.pre_tokenize(text)

        for original_words, sentence_tokens in self.text_to_tokens(text):
            raw_words, clean_words, clean_tokens = self._clean_tokens(
                sentence_tokens, replace_currency=self.do_replace_currency
            )

            # Don't yield empty sentences
            if raw_words or clean_words:

                # Use original words so whitespace is (mostly) retained
                raw_text = self.join_str.join(original_words)

                clean_text = self.join_str.join(clean_words)

                # Do post-processing
                post_clean_tokens = self.post_tokenize(clean_tokens)

                yield Sentence(
                    raw_text=raw_text,
                    raw_words=raw_words,
                    clean_text=clean_text,
                    clean_words=clean_words,
                    tokens=post_clean_tokens,
                )

    def post_tokenize(self, tokens: typing.Sequence[Token]) -> typing.Sequence[Token]:
        if self.pos_tagger is not None:
            # Predict tags for entire sentence
            pos_tags = self.pos_tagger([t.text for t in tokens])
            for token, pos in zip(tokens, pos_tags):
                token.features[TokenFeatures.PART_OF_SPEECH] = pos

        return tokens

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

    def match_number(
        self, text: str, number_converters: bool = False
    ) -> typing.Optional[REGEX_MATCH]:
        """
        Tries to determine if text is a number.

        Args:
            text: Text to try and match
            number_converters: If `True`, allow form "123_converter"

        Returns:
            Match if successful or None
        """
        if number_converters and self.number_converter_pattern:
            match = self.number_converter_pattern.match(text)
            if match is not None:
                return match

        if self.number_pattern:
            return self.number_pattern.match(text)

        return None

    # -------------------------------------------------------------------------

    def text_to_tokens(
        self, text: str
    ) -> typing.Iterable[typing.Tuple[typing.List[str], typing.List[Token]]]:
        """
        Process text into words and sentence tokens.

        Args:
            text: Text to process

        Returns:
            words and tokens for each sentence
        """
        # Sentence tokens have abbreviations expanded.
        original_words: typing.List[str] = []
        sentence_tokens: typing.List[Token] = []

        in_number = None
        for word_text in self.split_pattern.split(text):
            original_words.append(word_text)

            # Expand abbreviations
            # e.g., dr -> doctor
            word_text = self._expand_abbreviations(word_text)

            # Split again
            expanded_words = self.split_pattern.split(word_text)

            for exp_word_text in expanded_words:
                # Word or word with punctuation or currency symbol
                sub_words = [""]
                for i, c in enumerate(exp_word_text):
                    if (c in self.punctuations) or (c in self.currency_names):
                        # Punctuation or currency symbol
                        if in_number:
                            # Determine whether number is done
                            finish_number = False

                            if c in self.currency_names:
                                # <NUMBER> <CURRENCY>
                                finish_number = True
                            else:
                                # Peek forward to see if this is <NUMBER>.<NUMBER> or <NUMBER>.
                                if i < (len(exp_word_text) - 1):
                                    next_c = exp_word_text[i + 1]
                                    if not str.isdigit(next_c):
                                        # Next char is not a digit, so number stops here
                                        finish_number = True
                                else:
                                    # End of string after next char, so number can't continue
                                    finish_number = True

                            if finish_number:
                                sub_words.append("")
                                in_number = None

                        if in_number:
                            # Continue adding to number
                            sub_words[-1] += c
                        else:
                            # Start new sub-word
                            sub_words.append(c)
                            sub_words.append("")
                            in_number = None
                    else:
                        # Not a punctuation or currency symbol
                        sub_words[-1] += c
                        if str.isdigit(c):
                            if in_number is None:
                                # Start considering this sub-word a number
                                in_number = True
                        elif in_number:
                            # Stop considering this sub-word a number
                            in_number = False

                # Accumulate sub-words into full words
                for sub_word in sub_words:
                    if not sub_word:
                        # Skip empty sub-words
                        continue

                    # Append to current sentence
                    sentence_tokens.append(Token(text=sub_word))

                    if sub_word in self.major_breaks:
                        yield original_words, sentence_tokens

                        # New sentence
                        original_words = []
                        sentence_tokens = []

        if original_words or sentence_tokens:
            # Final sentence
            yield original_words, sentence_tokens

    def _expand_abbreviations(self, word: str) -> str:
        """Expand user-defined abbreviations"""
        # e.g., dr -> doctor
        if self.abbreviations:
            # Try to expand
            check_word = word
            if self.casing_func:
                # Fix case first
                check_word = self.casing_func(check_word)

            for pattern, to_text in self.abbreviations.items():
                new_word, num_subs = pattern.subn(to_text, check_word, count=1)
                if num_subs > 0:
                    return new_word

        return word

    def _clean_tokens(
        self, tokens: typing.Sequence[Token], replace_currency: bool = False
    ) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[Token]]:
        """
        Clean tokens and expand numbers.

        Returns: (raw_words, clean_words, clean_tokens)
        """
        raw_words: typing.List[str] = []
        clean_words: typing.List[str] = []
        clean_tokens: typing.List[Token] = []

        # Process each sentence
        last_token_currency: typing.Optional[str] = None
        last_token_was_break: bool = False

        # Process each token
        for token in tokens:
            token.text = token.text.strip()

            if not token.text:
                # Skip empty tokens
                continue

            raw_words.append(token.text)

            if (token.text in self.currency_names) and replace_currency:
                # Token will influence next number
                last_token_currency = token.text
                continue

            if (token.text in self.minor_breaks) or (token.text in self.major_breaks):
                # Keep breaks (pauses)
                if not last_token_was_break:
                    clean_words.append(token.text)
                    clean_tokens.append(token)

                    # Avoid multiple breaks
                    last_token_was_break = True

                continue

            last_token_was_break = False

            if (token.text in self.punctuations) or (
                self.exclude_non_words and (not self.is_word(token.text))
            ):
                # Skip non-words
                continue

            expanded_tokens = []

            if self.num2words_lang:
                # Try to convert number to words
                number_match = self.match_number(
                    token.text, number_converters=self.use_number_converters
                )
                if number_match:
                    expanded_tokens = self._try_expand_number(
                        number_match,
                        self.num2words_lang,
                        number_converters=self.use_number_converters,
                        replace_currency=self.do_replace_currency,
                        last_token_currency=last_token_currency,
                        babel_locale=self.babel_locale,
                    )

            if not expanded_tokens:
                # Not a number or number processing failed.
                # Process as a single word.
                expanded_tokens = [token]

            # Apply casing transformation
            if self.casing_func:
                for exp_token in expanded_tokens:
                    exp_token.text = self.casing_func(exp_token.text)

            clean_words.extend([t.text for t in expanded_tokens])
            clean_tokens.extend(expanded_tokens)

            last_token_currency = None

        return raw_words, clean_words, clean_tokens

    def _try_expand_number(
        self,
        number_match: REGEX_MATCH,
        num2words_lang: str,
        number_converters: bool = False,
        replace_currency: bool = False,
        num2words_currency: typing.Optional[str] = None,
        last_token_currency: typing.Optional[str] = None,
        babel_locale: typing.Optional[str] = None,
    ):
        """Attempt to convert a number to words using num2words and Babel."""
        match_groups = number_match.groups()

        try:
            assert match_groups, "Missing number group"

            digit_str = match_groups[0]
            num2words_kwargs = {"lang": num2words_lang}

            if number_converters and (len(match_groups) > 1):
                # Use converter name (second group in regex).
                # Available num2words converters are:
                # cardinal (default), ordinal, ordinal_num, year, currency
                digit_str, converter_str = match_groups

                if converter_str:
                    num2words_kwargs["to"] = converter_str

            if last_token_currency and replace_currency:
                currency_name = self.currency_names.get(last_token_currency)

                if currency_name:
                    # Last token was a currency symbol (e.g., '$')
                    num2words_kwargs["to"] = "currency"

                    # Add currency name
                    num2words_kwargs["currency"] = currency_name

            has_currency = num2words_kwargs.get("to") == "currency"

            if has_currency:
                if "currency" not in num2words_kwargs:
                    # Add language-specific currency (e.g., USD)
                    assert num2words_currency, "Default currency not provided"
                    num2words_kwargs["currency"] = num2words_currency

                # Custom separator so we can remove 'zero cents'
                num2words_kwargs["separator"] = "|"

            decimal_num: typing.Optional[Decimal] = None

            if babel_locale:
                try:
                    # Parse number according to locale using babel.
                    import babel
                    import babel.numbers

                    # This is important to handle thousand/decimal
                    # separators correctly.
                    decimal_num = babel.numbers.parse_decimal(
                        digit_str, locale=babel_locale
                    )
                except ImportError:
                    pass

            if decimal_num is None:
                # Parse with current locale
                decimal_num = Decimal(digit_str)

            # True if number has non-zero fractional part
            num_has_frac = (decimal_num % 1) != 0

            # num2words uses the number as an index sometimes, so it *has* to be
            # an integer, unless we're doing currency.
            if num_has_frac or has_currency:
                final_num = float(decimal_num)
            else:
                final_num = int(decimal_num)

            # Convert to words (e.g., 100 -> one hundred)
            num_str = num2words(final_num, **num2words_kwargs)

            if has_currency:
                # Post-process currency words
                num_str = self._post_process_currency(num_str, num_has_frac)

            # Remove all non-word characters
            num_str = re.sub(r"\W", " ", num_str).strip()

            # Tokenize number string itself
            number_words = self.split_pattern.split(num_str)

            # Successfully processed as a number
            return [Token(w) for w in number_words]
        except Exception:
            _LOGGER.exception(match_groups)

        # No tokens indicates failure to process as number
        return []

    def _post_process_currency(self, num_str: str, num_has_frac: bool) -> str:
        """Fix up num2words currency out and apply currency replacements."""
        if num_has_frac:
            # Discard num2words separator
            num_str = num_str.replace("|", "")
        else:
            # Remove 'zero cents' part
            num_str = num_str.split("|", maxsplit=1)[0]

        if self.currency_replacements:
            # Do regular expression replacements
            for pattern, replacement in self.currency_replacements:
                num_str = pattern.sub(replacement, num_str)

        return num_str

    def _make_abbreviation_patterns(
        self, abbreviations: typing.Mapping[typing.Union[str, REGEX_PATTERN], str]
    ) -> typing.MutableMapping[REGEX_PATTERN, str]:
        """Create regex patterns from abbrevations with optional surrounding punctuation"""
        punctuation_class = "[" + re.escape("".join(self.punctuations)) + "]"

        patterns: typing.MutableMapping[REGEX_PATTERN, str] = {}
        for from_text, to_text in abbreviations.items():
            if isinstance(from_text, REGEX_PATTERN):
                # Use literally
                patterns[from_text] = to_text
            elif self.punctuations:
                # Surround with optional punctuation
                pattern_text = (
                    f"^({punctuation_class}*)"
                    + "("
                    + re.escape(from_text)
                    + ")"
                    + f"({punctuation_class}*)$"
                )

                patterns[re.compile(pattern_text)] = f"\\1{to_text}\\3"
            else:
                # No punctuation
                pattern_text = "^" + re.escape(from_text) + "$"
                patterns[re.compile(pattern_text)] = to_text

        return patterns
