"""Class for cleaning raw text and tokenizing"""
import abc
import logging
import re
import typing
from decimal import Decimal

from num2words import num2words

from .const import REGEX_TYPE, Sentence, Token
from .utils import maybe_compile_regex

_LOGGER = logging.getLogger("gruut.toksen")

# -----------------------------------------------------------------------------


class Tokenizer(abc.ABC):
    """Base class for tokenizers"""

    def pre_tokenize(self, text: str) -> str:
        """Pre-process text before tokenization (called in tokenize)"""
        return text

    @abc.abstractmethod
    def tokenize(self, text: str, **kwargs) -> typing.Iterable[Sentence]:
        """Split text into tokenized sentences (with pre/post processing)"""
        pass

    def post_tokenize(self, tokens: typing.Sequence[Token]) -> typing.Sequence[Token]:
        """Post-process tokens (called in tokenize)"""
        return tokens


# -----------------------------------------------------------------------------


class RegexTokenizer(Tokenizer):
    """
    Full-featured tokenizer with support for number expansion.

    Pipline is (roughly):
    1. pre_tokenize applies regex replacements to raw text
    2. text is split using split_pattern into words
    3. words are split into sub-words using punctuations
       a. abbreviations are expanded
       b. sub-words are grouped by sentence and converted to Tokens
    4. Tokens are cleaned
       a. empty and non-word tokens are dropped
       b. numbers are expanded to words
       c. casing_func is applied

    Attributes
    ----------
    split_pattern: REGEX_TYPE
        str or Pattern used to split text into words.
        default: WHITESPACE_PATTERN

    join_str: str
        str used to join words back into text.
        default: " "

    replacements: Optional[Sequence[Tuple[REGEX_TYPE, str]]]
        Pattern, replacement tuples used in pre_tokenize on text.
        default: None

    casing_func: Optional[Callable[[str], str]]
        Function applied during token cleaning and abbreviation expansion (e.g., str.lower).
        default: None

    punctuations: Optional[Set[str]]
        Single-character strings that cause words to split into sub-words.
        Punctuation tokens are dropped from sentence tokens unless its a major/minor break.
        default: None

    minor_breaks: Optional[Set[str]]
        Single-character strings that indicate short pauses in a sentence.
        Minor break tokens are kept in sentence tokens.
        default: None

    major_breaks: Optional[Set[str]]
        Single-character strings that indicate the start of a new sentence.
        Major break tokens are kept in sentence tokens.
        default: None

    abbreviations: Optional[Mapping[str, Union[str, Sequence[str]]]]
        Short/long form mapping.
        Long form can be a sequence, and can therefore be multiple words.
        default: None

    drop_char_after_abbreviation: Optional[str]
        Single-character string that is automatically dropped after an abbreviation.
        Without this, "Dr." will become ["Dr", "."] and then ["Doctor", "."].
        Setting drop_char_after_abbreviation = "." will eliminate the trailing "." token.
        default: None

    number_pattern: REGEX_TYPE
        Pattern used to match numbers.
        First group must capture the number text.
        default: NUMBER_PATTERN

    number_converter_pattern: REGEX_TYPE
        Pattern used to match numbers with a converter specified.
        For example, 2021_year will expanded to "twenty twenty one" instead of "two thousand twenty one".
        First group must capture the number text, second group the converter.
        default: NUMBER_CONVERTER_PATTERN
        see also: use_number_converters

    non_word_pattern: Optional[REGEX_TYPE]
        Pattern used to match non-words, which may be excluded from sentence tokens.
        See also: exclude_non_words
        default: NON_WORD_PATTERN

    exclude_non_words: bool
        Excludes non words from sentence tokens (see RegexTokenizer.is_word).
        default: True

    num2words_lang: Optional[str]
        Language for num2words number expansion (e.g., "en_US").
        default: None

    babel_locale: Optional[str]
        Locale for babel number parsing (e.g., "en_US").
        default: None

    use_number_converters: bool
        If True, numbers may contain converters (see number_converter_pattern).
        default: False,

    currency_names: Optional[Dict[str, str]]
        Mapping from currency symbol (e.g., "$") to currency name (e.g., "USD").
        Used by num2words during number expansion.
        Currency symbols are also treated as punctuation during sub-word tokenization.
        default: None

    do_replace_currency: bool
        If True, numbers after a currency symbol are converted using the num2words "currency" converter.
        See currency_names.
        default: False

    currency_replacements: Optional[Sequence[Tuple[REGEX_TYPE, str]]]
        Pattern, replacement tuples that are applied to the string returned by num2words after currency conversion.
        Example: $1.50 becomes "one dollar, fifty cents", so you may want to replace "," with " and ".
        default: None
    """

    # Pattern for initially splitting text into words
    WHITESPACE_PATTERN = re.compile(r"\s+")

    # Default pattern for matching non-words
    NON_WORD_PATTERN = re.compile(r"^(\W|_)+$")

    # Default pattern for matching numbers like 3.14.
    # First group must be number text ("3.14").
    NUMBER_PATTERN = re.compile(r"^(-?\d+(?:[,.]\d+)*)$")

    # Default pattern for matching numbers with a "converter" like "1970_year".
    # First group must be number text ("1970").
    # Second group must be converter ("year").
    NUMBER_CONVERTER_PATTERN = re.compile(r"^(-?\d+(?:[,.]\d+)*)_(\w+)$")

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
            typing.Mapping[str, typing.Union[str, typing.Sequence[str]]]
        ] = None,
        drop_char_after_abbreviation: typing.Optional[str] = None,
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
        self.abbreviations = abbreviations or {}
        self.drop_char_after_abbreviation = drop_char_after_abbreviation
        self.number_pattern = number_pattern
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

    def pre_tokenize(self, text: str) -> str:
        """Pre-process text before tokenization (called in tokenize)"""
        for pattern, replacement in self.replacements:
            text = pattern.sub(replacement, text)

        return text

    def tokenize(self, text: str, **kwargs) -> typing.Iterable[Sentence]:
        """Split text into tokenized sentences (with pre/post processing)"""
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
                clean_tokens = self.post_tokenize(clean_tokens)

                yield Sentence(
                    raw_text=raw_text,
                    raw_words=raw_words,
                    clean_text=clean_text,
                    clean_words=clean_words,
                    tokens=clean_tokens,
                )

    def is_word(self, text: str) -> bool:
        """True if text is considered a word"""
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
    ) -> typing.Optional[re.Match]:
        """Returns a regex Match if text is a number"""
        if number_converters and self.number_converter_pattern:
            return self.number_converter_pattern.match(text)

        if self.number_pattern:
            return self.number_pattern.match(text)

        return None

    # -------------------------------------------------------------------------

    def text_to_tokens(
        self, text: str
    ) -> typing.Iterable[typing.Tuple[typing.List[str], typing.List[Token]]]:
        """
        Process text into words and sentence tokens.

        Returns: (original_words, sentence_tokens) for each sentence
        """
        # Sentence tokens have abbreviations expanded.
        original_words: typing.List[str] = []
        sentence_tokens: typing.List[Token] = []

        in_number = None
        for word_text in self.split_pattern.split(text):
            original_words.append(word_text)

            # Word or word with punctuation or currency symbol
            sub_words = [""]
            for i, c in enumerate(word_text):
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
                            if i < (len(word_text) - 1):
                                next_c = word_text[i + 1]
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
            last_word_was_abbreviation = False
            for sub_word_idx, sub_word in enumerate(sub_words):
                if not sub_word:
                    # Skip empty sub-words
                    continue

                if last_word_was_abbreviation and (
                    sub_word == self.drop_char_after_abbreviation
                ):
                    # Skip symbol after abbreviation
                    # e.g., dr. -> dr
                    continue

                # Expand abbreviations
                # e.g., dr -> doctor
                expanded_words = []
                if self.abbreviations:
                    # Try to expand
                    check_word = sub_word
                    if self.casing_func:
                        # Fix case first
                        check_word = self.casing_func(check_word)

                    # Expansions may be multiple words.
                    expansion = self.abbreviations.get(check_word)
                    if expansion:
                        if isinstance(expansion, str):
                            # Single word
                            expanded_words.append(expansion)
                        else:
                            # Multiple words
                            expanded_words.extend(expansion)

                        last_word_was_abbreviation = True
                    else:
                        # Not an abbreviation (no expansion)
                        expanded_words.append(check_word)
                else:
                    # No abbreviations, use word as-is
                    expanded_words.append(sub_word)

                # Append to current sentence
                for exp_word in expanded_words:
                    sentence_tokens.append(Token(text=exp_word))

                    if exp_word in self.major_breaks:
                        yield original_words, sentence_tokens

                        # New sentence
                        original_words = []
                        sentence_tokens = []

        if original_words or sentence_tokens:
            # Final sentence
            yield original_words, sentence_tokens

    def _clean_tokens(
        self, tokens: typing.Sequence[Token], replace_currency: bool = False
    ) -> typing.Tuple[typing.List[str], typing.List[str], typing.List[Token]]:
        """
        Clean tokens and expand numbers.

        Returns: raw_words, clean_words, clean_tokens
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
        number_match: re.Match,
        num2words_lang: str,
        number_converters: bool = False,
        replace_currency: bool = False,
        num2words_currency: typing.Optional[str] = None,
        last_token_currency: typing.Optional[str] = None,
        babel_locale: typing.Optional[str] = None,
    ):
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
        except Exception as e:
            _LOGGER.exception(match_groups)

        # No tokens indicates failure to process as number
        return []

    def _post_process_currency(self, num_str: str, num_has_frac: bool) -> str:
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


# -----------------------------------------------------------------------------


# class Tokenizer:
#     """Splits text into sentences, tokenizes and cleans"""

#     def __init__(
#         self,
#         config,
#         custom_tokenize: typing.Optional[TokenizeFunc] = None,
#         custom_post_tokenize: typing.Optional[PostTokenizeFunc] = None,
#     ):
#         self.config = config
#         self.language = pydash.get(self.config, "language.code")

#         self.custom_tokenize = custom_tokenize
#         self.custom_post_tokenize = custom_post_tokenize

#         # Symbol to skip immediately after an abbreviation
#         self.abbreviation_skip = pydash.get(self.config, "symbols.abbreviation_skip")

#         # Short pause symbols (commas, etc.)
#         self.minor_breaks: typing.Set[str] = set(
#             pydash.get(self.config, "symbols.minor_breaks", [])
#         )

#         # End of sentence symbols
#         self.major_breaks: typing.Set[str] = set(
#             pydash.get(self.config, "symbols.major_breaks", [])
#         )

#         # If True, keep question marks
#         self.question_mark = bool(
#             pydash.get(self.config, "symbols.question_mark", False)
#         )

#         # Regex to split words
#         self.token_split_pattern = re.compile(
#             pydash.get(self.config, "symbols.token_split", r"\s+")
#         )

#         # String to join words
#         self.token_join = pydash.get(self.config, "symbols.token_join", " ")

#         # Characters that will cause breaks within words
#         self.punctuations: typing.Set[str] = set(
#             pydash.get(self.config, "symbols.punctuations", [])
#         )

#         # Regex to match numbers (digits)
#         self.number_pattern: typing.Optional[re.Pattern] = None
#         self.number_converter_pattern: typing.Optional[re.Pattern] = None

#         number_regex = pydash.get(self.config, "symbols.number_regex")
#         if number_regex:
#             self.number_pattern = re.compile(number_regex)

#             # Slip _converter into the regex
#             if number_regex[-1] == "$":
#                 # Tuck behind end of expression $
#                 number_converter_regex = number_regex[:-1] + r"\w+$"
#             else:
#                 # Append directly
#                 number_converter_regex = number_regex + r"\w+$"

#             _LOGGER.debug("Number converter regex: %s", number_converter_regex)
#             self.number_converter_pattern = re.compile(number_converter_regex)

#         babel_locale_str = pydash.get(self.config, "symbols.babel_locale")

#         if not babel_locale_str:
#             # en-us -> en_US
#             locale_parts = self.language.split("-", maxsplit=1)

#             if len(locale_parts) < 2:
#                 babel_locale_str = locale_parts[0]
#             else:
#                 babel_locale_str = (
#                     locale_parts[0].lower() + "_" + locale_parts[1].upper()
#                 )

#         self.babel_locale_str = babel_locale_str
#         self.babel_locale = babel.Locale(self.babel_locale_str)

#         self.num2words_lang = pydash.get(
#             self.config, "symbols.num2words_lang", self.language
#         )

#         # Default currency to use (e.g., USD)
#         self.currency: typing.Optional[str] = pydash.get(
#             self.config, "numbers.currency"
#         )

#         # Build a map of currency symbols to currency names for the locale
#         self.currency_names = {
#             babel.numbers.get_currency_symbol(cn): cn
#             for cn in self.babel_locale.currency_symbols
#         }

#         # Case transformation (lower/upper)
#         casing = pydash.get(self.config, "symbols.casing")
#         self.casing: typing.Optional[typing.Callable[[str], str]] = None
#         if casing == "lower":
#             self.casing = str.lower
#         elif casing == "upper":
#             self.casing = str.upper

#         # [(pattern, replacement)]
#         self.replacements = []
#         for replace_key, replace_value in pydash.get(
#             self.config, "symbols.replace", {}
#         ).items():
#             self.replacements.append((re.compile(replace_key), replace_value))

#         # short form -> [expansion words]
#         self.abbreviations: typing.Dict[str, typing.List[str]] = {}
#         for abbrev_key, abbrev_value in self.config.get("abbreviations", {}).items():
#             if isinstance(abbrev_value, str):
#                 # One word expansion
#                 abbrev_value = [abbrev_value]

#             # short form -> [expansion words]
#             self.abbreviations[abbrev_key] = abbrev_value

#     # -------------------------------------------------------------------------

#     def tokenize(
#         self,
#         text: str,
#         number_converters: bool = False,
#         replace_currency: bool = True,
#         guess_pos: bool = True,
#     ) -> typing.Iterable[Sentence]:
#         """Split text into sentences, tokenize, and clean"""
#         sentence_tokens: typing.List[typing.List[Token]] = []
#         raw_sentence_tokens: typing.List[typing.List[str]] = []

#         if self.custom_tokenize:
#             # User-defined tokenization
#             sentence_tokens = self.custom_tokenize(text, guess_pos=guess_pos)
#             raw_sentence_tokens = [[t.text for t in s] for s in sentence_tokens]
#         else:
#             # Do pre-tokenization replacements
#             for pattern, replacement in self.replacements:
#                 text = pattern.sub(replacement, text)

#             # Tokenize
#             raw_tokens = self.token_split_pattern.split(text)

#             # Break raw tokens into sentences and sub-tokens according to
#             # punctuation.
#             # Performance is going to be bad, but this is a first pass.
#             raw_sentence_tokens.append([])
#             sentence_tokens.append([])

#             in_number = None
#             for token_text in raw_tokens:
#                 raw_sentence_tokens[-1].append(token_text)

#                 # Word or word with punctuation or currency symbol
#                 sub_tokens = [""]
#                 for i, c in enumerate(token_text):
#                     if (c in self.punctuations) or (c in self.currency_names):
#                         if in_number:
#                             # Determine whether number is done
#                             finish_number = False

#                             if c in self.currency_names:
#                                 # <NUMBER> <CURRENCY>
#                                 finish_number = True
#                             else:
#                                 # Peek forward to see if this is <NUMBER>.<NUMBER> or <NUMBER>.
#                                 if i < (len(token_text) - 1):
#                                     next_c = token_text[i + 1]
#                                     if not str.isdigit(next_c):
#                                         # Next char is not a digit, so number stops here
#                                         finish_number = True
#                                 else:
#                                     # End of string after next char, so number can't continue
#                                     finish_number = True

#                             if finish_number:
#                                 sub_tokens.append("")
#                                 in_number = None

#                         if in_number:
#                             # Continue adding to number
#                             sub_tokens[-1] += c
#                         else:
#                             # Start new sub-token
#                             sub_tokens.append(c)
#                             sub_tokens.append("")
#                     else:
#                         sub_tokens[-1] += c
#                         if str.isdigit(c):
#                             if in_number is None:
#                                 in_number = True
#                         else:
#                             in_number = False

#                 # Accumulate sub-tokens into sentence tokens
#                 last_token_was_abbreviation = False
#                 for sub_token in sub_tokens:
#                     if not sub_token:
#                         continue

#                     if last_token_was_abbreviation and (
#                         sub_token == self.abbreviation_skip
#                     ):
#                         # Skip period after abbreviation
#                         continue

#                     # Expand abbreviations
#                     expanded_tokens = []
#                     if self.abbreviations:
#                         check_token = sub_token
#                         if self.casing:
#                             check_token = self.casing(sub_token)

#                         # Expansions may be multiple words.
#                         expansion = self.abbreviations.get(check_token)
#                         if expansion:
#                             expanded_tokens.extend(expansion)
#                             last_token_was_abbreviation = True
#                         else:
#                             expanded_tokens.append(check_token)
#                     else:
#                         expanded_tokens.append(sub_token)

#                     # Append to current sentence
#                     for ex_token in expanded_tokens:
#                         sentence_tokens[-1].append(Token(text=ex_token))

#                         if ex_token in self.major_breaks:
#                             # New sentence
#                             sentence_tokens.append([])
#                             raw_sentence_tokens.append([])

#         # Process each sentence
#         last_token_currency: typing.Optional[str] = None
#         last_token_was_break: bool = False

#         for sentence_idx, sentence in enumerate(sentence_tokens):
#             raw_words = []
#             clean_words = []
#             clean_tokens = []

#             # Process each token
#             for token in sentence:
#                 token.text = token.text.strip()

#                 if not token.text:
#                     # Skip empty tokens
#                     continue

#                 raw_words.append(token.text)

#                 if (token.text in self.currency_names) and replace_currency:
#                     # Token will influence next number
#                     last_token_currency = token.text
#                     continue

#                 if (token.text in self.minor_breaks) or (
#                     token.text in self.major_breaks
#                 ):
#                     # Keep breaks (pauses)
#                     if not last_token_was_break:
#                         clean_words.append(token.text)
#                         clean_tokens.append(token)

#                         # Avoid multiple breaks
#                         last_token_was_break = True

#                     continue

#                 last_token_was_break = False

#                 if (token.text in self.punctuations) or (_NON_WORD.match(token.text)):
#                     # Skip non-words
#                     continue

#                 process_as_word = True

#                 # Try to process as a number first
#                 number_match = None
#                 if number_converters and self.number_converter_pattern:
#                     number_match = self.number_converter_pattern.match(token.text)
#                 elif self.number_pattern:
#                     number_match = self.number_pattern.match(token.text)

#                 if number_match:
#                     try:
#                         digit_str = token.text
#                         num2words_kwargs = {"lang": self.num2words_lang}

#                         if number_converters:
#                             # Look for 123_converter pattern.
#                             # Available num2words converters are:
#                             # cardinal (default), ordinal, ordinal_num, year, currency
#                             digit_str, converter_str = token.text.split("_", maxsplit=1)

#                             if converter_str:
#                                 num2words_kwargs["to"] = converter_str

#                         if last_token_currency and replace_currency:
#                             # Last token was a currency symbol (e.g., '$')
#                             num2words_kwargs["to"] = "currency"

#                             # Add currency name
#                             num2words_kwargs["currency"] = self.currency_names.get(
#                                 last_token_currency, self.currency
#                             )

#                         has_currency = num2words_kwargs.get("to") == "currency"

#                         if has_currency and self.currency:
#                             if "currency" not in num2words_kwargs:
#                                 # Add language-specific currency (e.g., USD)
#                                 num2words_kwargs["currency"] = self.currency

#                             # Custom separator so we can remove 'zero cents'
#                             num2words_kwargs["separator"] = "|"

#                         # Parse number according to locale.
#                         # This is important to handle thousand/decimal
#                         # separators correctly.
#                         num = babel.numbers.parse_decimal(
#                             digit_str, locale=self.babel_locale_str
#                         )

#                         # True if number has non-zero fractional part
#                         num_has_frac = (num % 1) != 0

#                         if not num_has_frac:
#                             # num2words uses the number as an index sometimes,
#                             # so it *has* to be an integer, unless we're doing
#                             # currency.
#                             if has_currency:
#                                 num = float(num)
#                             else:
#                                 num = int(num)

#                         # Convert to words (e.g., 100 -> one hundred)
#                         num_str = num2words(num, **num2words_kwargs)

#                         if has_currency:

#                             if num_has_frac:
#                                 # Discard separator
#                                 num_str = num_str.replace("|", "")
#                             else:
#                                 # Remove 'zero cents' part
#                                 num_str = num_str.split("|", maxsplit=1)[0]

#                         # Remove all non-word characters
#                         num_str = re.sub(r"\W", " ", num_str).strip()

#                         # Tokenize number string itself
#                         num_tokens = self.token_split_pattern.split(num_str)

#                         if self.casing:
#                             # Apply casing transformation
#                             num_tokens = [self.casing(t) for t in num_tokens]

#                         clean_words.extend(num_tokens)
#                         clean_tokens.extend([Token(nt) for nt in num_tokens])

#                         # Successfully processed as a number
#                         process_as_word = False
#                     except Exception:
#                         _LOGGER.exception(token)

#                 if process_as_word:
#                     # Not a number
#                     words = [token]

#                     # Apply casing transformation
#                     if self.casing:
#                         for word in words:
#                             word.text = self.casing(word.text)

#                     clean_words.extend([w.text for w in words])
#                     clean_tokens.extend(words)

#                 last_token_currency = None

#             # -----------------------------------------------------------------

#             # Use raw sentence tokens from first stage so whitespace is (mostly) retained
#             raw_text = self.token_join.join(raw_sentence_tokens[sentence_idx])

#             # Don't yield empty sentences
#             if raw_words or clean_words:

#                 # Do post-processing
#                 if self.custom_post_tokenize:
#                     clean_tokens = self.custom_post_tokenize(
#                         clean_tokens, guess_pos=guess_pos
#                     )

#                 yield Sentence(
#                     raw_text=raw_text,
#                     raw_words=raw_words,
#                     clean_words=clean_words,
#                     tokens=clean_tokens,
#                 )

#     # -------------------------------------------------------------------------

#     def is_word(self, word: str) -> bool:
#         """True if word is not empty, a break, or punctuation"""
#         word = word.strip()
#         return (
#             len(word) > 0
#             and (word not in self.minor_breaks)
#             and (word not in self.major_breaks)
#             and (word not in self.punctuations)
#         )
