"""
Language-specific tokenizers and phonemizers for gruut.

The :py:meth:`~gruut.lang.get_tokenizer` and :py:meth:`~gruut.lang.get_phonemizer` methods will get the appropriate tokenizer/phonemizer for a known language.
See :py:meth:`~gruut.lang.resolve_lang` to understand how language aliases are resolved.
"""
import itertools
import logging
import typing
from pathlib import Path

from gruut_ipa import IPA, Phonemes

from .const import TOKEN_OR_STR, WORD_PHONEMES, Token, TokenFeatures, WordPronunciation
from .phonemize import Phonemizer, SqlitePhonemizer
from .toksen import RegexTokenizer, Tokenizer
from .utils import find_lang_dir, get_currency_names, pairwise

_LOGGER = logging.getLogger("gruut.lang")

# -----------------------------------------------------------------------------

LANG_ALIASES = {
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


def resolve_lang(lang: str) -> str:
    """
    Try to resolve language using aliases.

    Args:
        lang: Language name or alias

    Returns:
        Resolved language name
    """
    lang = LANG_ALIASES.get(lang, lang)

    if lang not in KNOWN_LANGS:
        # Try with _ replaced by -
        maybe_lang = lang.replace("_", "-")
        if maybe_lang in KNOWN_LANGS:
            lang = maybe_lang

    return lang


def get_tokenizer(
    lang: str,
    lang_dir: typing.Optional[typing.Union[str, Path]] = None,
    no_pos: bool = False,
    **kwargs,
) -> Tokenizer:
    """
    Get language-specific tokenizer.

    Args:
        lang: Language name or alias
        lang_dir: Optional path to language files directory (see also :py:meth:`~gruut.utils.find_lang_dir`)
        no_pos: If ``True``, part of speech tagging is disabled for supported tokenizers
        kwargs: Keyword arguments passed to tokenizer's ``__init__`` method

    Returns:
        Language-specific tokenizer
    """
    lang = resolve_lang(lang)

    if (lang_dir is None) and (lang in KNOWN_LANGS):
        lang_dir = find_lang_dir(lang)

    if lang_dir is not None:
        lang_dir = Path(lang_dir)

        if "pos_model" not in kwargs:
            # Use part of speech tagger in model directory (optional)
            pos_model = lang_dir / "pos" / "model.crf"
            if pos_model.is_file():
                kwargs["pos_model"] = pos_model

    if no_pos:
        # Don't use part-of-speech tagger
        kwargs.pop("pos_model", None)

    if lang == "cs-cz":
        assert lang_dir is not None
        return CzechTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "de-de":
        assert lang_dir is not None
        return GermanTokenizer(lang_dir=lang_dir, **kwargs)

    if lang in ENGLISH_LANGS:
        assert lang_dir is not None
        return EnglishTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "es-es":
        assert lang_dir is not None
        return SpanishTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "fa":
        assert lang_dir is not None
        return FarsiTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "fr-fr":
        assert lang_dir is not None
        return FrenchTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "it-it":
        assert lang_dir is not None
        return ItalianTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "nl":
        assert lang_dir is not None
        return DutchTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "pt":
        assert lang_dir is not None
        return PortugueseTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "ru-ru":
        assert lang_dir is not None
        return RussianTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "sv-se":
        assert lang_dir is not None
        return SwedishTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "sw":
        assert lang_dir is not None
        return SwahiliTokenizer(lang_dir=lang_dir, **kwargs)

    # Fall back to basic regex tokenizer
    return RegexTokenizer(**kwargs)


def get_phonemizer(
    lang: str,
    lang_dir: typing.Optional[typing.Union[str, Path]] = None,
    model_prefix: typing.Optional[str] = None,
    no_g2p: bool = False,
    fr_no_liason: bool = False,
    **kwargs,
) -> Phonemizer:
    """
    Get language-specific phonemizer.

    Args:
        lang: Language name or alias
        lang_dir: Optional path to language files directory (see also :py:meth:`~gruut.utils.find_lang_dir`)
        model_prefix: Optional directory prefix used for lexicon.db and g2p/model.crf
        no_g2p: If ``True``, disable grapheme to phoneme prediction for unknown words
        fr_no_liason: If ``True``, disable addition of liasons in :py:class:`~gruut.lang.FrenchPhonemizer`
        kwargs: Keyword arguments passed to phonemizer's ``__init__`` method

    Returns:
        Language-specific phonemizer
    """
    lang = resolve_lang(lang)

    if (lang_dir is None) and (lang in KNOWN_LANGS):
        lang_dir = find_lang_dir(lang)

    if lang_dir is not None:
        lang_dir = Path(lang_dir)

    if lang in KNOWN_LANGS:
        assert lang_dir is not None
        prefix_dir = lang_dir

        if model_prefix:
            prefix_dir = prefix_dir / model_prefix

        if "database_path" not in kwargs:
            # Use database in model directory (required)
            kwargs["database_path"] = str(prefix_dir / "lexicon.db")

        if "g2p_model" not in kwargs:
            # Use grapheme to phoneme model in model directory (optional)
            g2p_model = prefix_dir / "g2p" / "model.crf"
            if g2p_model.is_file():
                kwargs["g2p_model"] = g2p_model

    if no_g2p:
        # Don't use grapheme-to-phoneme model
        kwargs.pop("g2p_model", None)

    if lang == "cs-cz":
        assert lang_dir is not None
        return CzechPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "de-de":
        assert lang_dir is not None
        return GermanPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang in ENGLISH_LANGS:
        assert lang_dir is not None
        return EnglishPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "es-es":
        assert lang_dir is not None
        return SpanishPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "fa":
        assert lang_dir is not None
        return FarsiPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "fr-fr":
        assert lang_dir is not None
        return FrenchPhonemizer(lang_dir=lang_dir, no_liason=fr_no_liason, **kwargs)

    if lang == "it-it":
        assert lang_dir is not None
        return ItalianPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "nl":
        assert lang_dir is not None
        return DutchPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "pt":
        assert lang_dir is not None
        return PortuguesePhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "ru-ru":
        assert lang_dir is not None
        return RussianPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "sv-se":
        assert lang_dir is not None
        return SwedishPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "sw":
        assert lang_dir is not None
        return SwahiliPhonemizer(lang_dir=lang_dir, **kwargs)

    # Fall back to basic sqlite phonemizer.
    # This will fail if no database_path argument is provided.
    return SqlitePhonemizer(**kwargs)


# -----------------------------------------------------------------------------


def id_to_phonemes(
    lang: str,
    lang_phonemes: typing.Optional[typing.Iterable[str]] = None,
    pad: str = "_",
    no_pad: bool = False,
    no_word_break: bool = False,
    no_stress: bool = False,
    no_accents: typing.Optional[bool] = None,
    tones: typing.Optional[typing.Iterable[str]] = None,
) -> typing.Sequence[str]:
    """
    Create an ordered list of phonemes for a language.

    Useful for transforming phoneme strings into tensors.

    Args:
        lang: Language name or alias
        lang_phonemes: Ordered phonemes for a language or None if phonemes from :py:mod:`gruut_ipa` should be used
        pad: String used for empty first phoneme (index 0)
        no_pad: If ``True``, don't include pad phoneme
        no_word_break: If ``True``, don't include IPA word break phoneme
        no_stress: If ``True``, don't include IPA minor/major break phonemes
        no_accents: If ``True``, don't include IPA accent phonemes. If ``None``, decide based on ``lang``
        tones: Optional ordered list of language tones phonemes

    Returns:
        Ordered sequence of phonemes for language
    """

    lang = resolve_lang(lang)

    if lang_phonemes is None:
        # Use gruut-ipa for phonemes list
        lang_phonemes = [p.text for p in Phonemes.from_language(lang)]

    assert lang_phonemes is not None

    if no_accents is None:
        # Only add accents for Swedish
        no_accents = lang != "sv-se"

    # Acute/grave accents (' and ²)
    accents = []
    if not no_accents:
        # Accents from Swedish, etc.
        accents = [IPA.ACCENT_ACUTE.value, IPA.ACCENT_GRAVE.value]

    # Primary/secondary stress (ˈ and ˌ)
    # NOTE: Accute accent (0x0027) != primary stress (0x02C8)
    stresses = []
    if not no_stress:
        stresses = [IPA.STRESS_PRIMARY.value, IPA.STRESS_SECONDARY.value]

    # Tones
    tones = list(tones) if tones is not None else []

    # Word break
    word_break = []
    if not no_word_break:
        word_break = [IPA.BREAK_WORD.value]

    # Pad symbol must always be first (index 0)
    phonemes_list = []
    if not no_pad:
        phonemes_list.append(pad)

    # Order here is critical
    phonemes_list = (
        phonemes_list
        + [IPA.BREAK_MINOR.value, IPA.BREAK_MAJOR.value]
        + word_break
        + accents
        + stresses
        + tones
        + sorted(list(lang_phonemes))
    )

    return phonemes_list


# -----------------------------------------------------------------------------
# cs-cz
# -----------------------------------------------------------------------------

CZECH_MINOR_BREAKS = {",", ":", ";"}
CZECH_MAJOR_BREAKS = {".", "?", "!"}


class CzechTokenizer(RegexTokenizer):
    """Tokenizer for Czech (čeština)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("cs_CZ")
        currency_names["€"] = "EUR"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={'"', ",", ";", ":", ".", "?", "!", "„", "“", "”", "«", "»"},
            minor_breaks=CZECH_MINOR_BREAKS,
            major_breaks=CZECH_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="cs_CZ",
            babel_locale="cs_CZ",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class CzechPhonemizer(SqlitePhonemizer):
    """Phonemizer for Czech (čeština)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=CZECH_MINOR_BREAKS, major_breaks=CZECH_MAJOR_BREAKS, **kwargs
        )


# -----------------------------------------------------------------------------
# de-de
# -----------------------------------------------------------------------------

GERMAN_MINOR_BREAKS = {",", ":", ";"}
GERMAN_MAJOR_BREAKS = {".", "?", "!"}


class GermanTokenizer(RegexTokenizer):
    """Tokenizer for German (Deutsch)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("de_DE")
        currency_names["€"] = "EUR"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
            ],
            punctuations={
                '"',
                ",",
                ";",
                ":",
                ".",
                "?",
                "!",
                "„",
                "“",
                "”",
                "«",
                "»",
                "’",
            },
            minor_breaks=GERMAN_MINOR_BREAKS,
            major_breaks=GERMAN_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="de_DE",
            babel_locale="de_DE",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class GermanPhonemizer(SqlitePhonemizer):
    """Phonemizer for German (Deutsch)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=GERMAN_MINOR_BREAKS, major_breaks=GERMAN_MAJOR_BREAKS, **kwargs
        )


# -----------------------------------------------------------------------------
# en-us, en-gb
# -----------------------------------------------------------------------------

ENGLISH_PUNCTUATIONS = {'"', ",", ";", ":", ".", "?", "!", "“", "”", "«", "»", "-"}
ENGLISH_MINOR_BREAKS = {",", ":", ";"}
ENGLISH_MAJOR_BREAKS = {".", "?", "!"}


class EnglishTokenizer(RegexTokenizer):
    """Tokenizer for English"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("en_US")
        currency_names["$"] = "USD"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("&", " and "),
                ("’", "'"),  # normalize apostrophe
                ("(\\d+(?:[,.]\\d+)*)%\\B", "\\1 percent"),
            ],
            abbreviations={
                "capt.": "captain",
                "co.": "company",
                "col.": "colonel",
                "dr.": "doctor",
                "drs.": "doctors",
                "esq.": "esquire",
                "ft.": "fort",
                "gen.": "general",
                "hon.": "honorable",
                "jr.": "junior",
                "ltd.": "limited",
                "lt.": "lieutenant",
                "maj.": "major",
                "mr.": "mister",
                "mrs.": "misess",
                "rev.": "reverend",
                "sgt.": "sergeant",
                "st.": "saint",
            },
            punctuations=ENGLISH_PUNCTUATIONS,
            minor_breaks=ENGLISH_MINOR_BREAKS,
            major_breaks=ENGLISH_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="en_US",
            babel_locale="en_US",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class EnglishPhonemizer(SqlitePhonemizer):
    """Phonemizer for English"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=ENGLISH_MINOR_BREAKS,
            major_breaks=ENGLISH_MAJOR_BREAKS,
            feature_map={
                TokenFeatures.PART_OF_SPEECH: {
                    "NNS": "NN",
                    "NNP": "NN",
                    "NNPS": "NN",
                    "PRP$": "PRP",
                    "RBR": "RB",
                    "RBS": "RB",
                    "VBG": "VB",
                    "VBN": "VB",
                    "VBP": "VB",
                    "VBZ": "VB",
                    "JJR": "JJ",
                    "JJS": "JJ",
                }
            },
            **kwargs,
        )


# -----------------------------------------------------------------------------
# es-es
# -----------------------------------------------------------------------------

SPANISH_MINOR_BREAKS = {",", ":", ";"}
SPANISH_MAJOR_BREAKS = {".", "?", "!"}


class SpanishTokenizer(RegexTokenizer):
    """Tokenizer for Spanish (Español)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("es_ES")
        currency_names["€"] = "EUR"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={
                '"',
                ",",
                ";",
                ":",
                ".",
                "?",
                "¿",
                "!",
                "¡",
                "„",
                "“",
                "”",
                "«",
                "»",
            },
            minor_breaks=SPANISH_MINOR_BREAKS,
            major_breaks=SPANISH_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="es_ES",
            babel_locale="es_ES",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class SpanishPhonemizer(SqlitePhonemizer):
    """Phonemizer for Spanish (Español)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=SPANISH_MINOR_BREAKS,
            major_breaks=SPANISH_MAJOR_BREAKS,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# fa
# -----------------------------------------------------------------------------


class FarsiTokenizer(RegexTokenizer):
    """Tokenizer for Farsi/Persian (فارسی)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)
        currency_names = get_currency_names("fa")

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
            ],
            punctuations={
                '"',
                "„",
                "“",
                "”",
                "«",
                "»",
                "’",
                ",",
                "،",
                ":",
                ";",
                ".",
                "?",
                "؟",
                "!",
            },
            minor_breaks={"،", ":", ";"},
            major_breaks={".", "؟", "!"},
            casing_func=str.lower,
            num2words_lang="fa",
            babel_locale="fa",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )

    def text_to_tokens(
        self, text: str
    ) -> typing.Iterable[typing.Tuple[typing.List[str], typing.List[Token]]]:
        """
        Process text into words and sentence tokens using hazm.

        Returns: (original_words, sentence_tokens) for each sentence
        """

        try:
            import hazm
        except ImportError:
            _LOGGER.warning("hazm is highly recommended for language 'fa'")
            _LOGGER.warning("pip install 'hazm>=0.7.0'")

            # Fall back to parent implementation
            yield from super().text_to_tokens(text)

        # Load normalizer
        if not hasattr(self, "normalizer"):
            normalizer = hazm.Normalizer()
            setattr(self, "normalizer", normalizer)

        # Load tagger
        if not hasattr(self, "tagger"):
            # Load part of speech tagger
            model_path = self.lang_dir / "postagger.model"
            tagger = hazm.POSTagger(model=str(model_path))
            setattr(self, "tagger", tagger)

        sentences = hazm.sent_tokenize(normalizer.normalize(text))
        for sentence in sentences:
            original_words = []
            sentence_tokens = []
            for word, pos in tagger.tag(hazm.word_tokenize(sentence)):
                original_words.append(word)
                sentence_tokens.append(
                    Token(text=word, features={TokenFeatures.PART_OF_SPEECH: pos})
                )

            yield original_words, sentence_tokens


class FarsiPhonemizer(SqlitePhonemizer):
    """Phonemizer for Farsi/Persian (فارسی)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks={"،", ":", ";"}, major_breaks={".", "؟", "!"}, **kwargs
        )

    def post_phonemize(
        self, token: Token, token_pron: WordPronunciation
    ) -> WORD_PHONEMES:
        """Post-process tokens/pronunciton after phonemization (called in phonemize)"""
        phonemes = super().post_phonemize(token, token_pron)

        # Genitive case
        pos = token.features.get(TokenFeatures.PART_OF_SPEECH)
        if pos == "Ne":
            if isinstance(phonemes, list):
                phonemes.append("e̞")
            else:
                return list(phonemes) + ["e̞"]

        return phonemes


# -----------------------------------------------------------------------------
# fr-fr
# -----------------------------------------------------------------------------

FRENCH_MINOR_BREAKS = {",", ":", ";"}
FRENCH_MAJOR_BREAKS = {".", "?", "!"}


class FrenchTokenizer(RegexTokenizer):
    """Tokenizer for French (Français)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("fr_FR")
        currency_names["€"] = "EUR"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            abbreviations={
                "M.": "monsieur",
                "Mlle.": "mademoiselle",
                "Mlles.": "mesdemoiselles",
                "Mme.": "Madame",
                "Mmes.": "Mesdames",
                "N.B.": "nota bene",
                "p.c.q.": "parce que",
                "Pr.": "professeur",
                "qqch.": "quelque chose",
                "rdv.": "rendez-vous",
                "max.": "maximum",
                "min.": "minimum",
                "no.": "numéro",
                "adr.": "adresse",
                "dr.": "docteur",
                "st.": "saint",
                "co.": "companie",
                "jr.": "junior",
                "sgt.": "sergent",
                "capt.": "capitain",
                "col.": "colonel",
                "av.": "avenue",
                "av. J.-C.": "avant Jésus-Christ",
                "apr. J.-C.": "après Jésus-Christ",
                "art.": "article",
                "boul.": "boulevard",
                "c.-à-d.": "c’est-à-dire",
                "etc.": "et cetera",
                "ex.": "exemple",
                "excl.": "exclusivement",
                "Mlle": "mademoiselle",
                "Mlles": "mesdemoiselles",
                "Mme": "Madame",
                "Mmes": "Mesdames",
            },
            punctuations={'"', ",", ";", ":", ".", "?", "!", "„", "“", "”", "«", "»"},
            minor_breaks=FRENCH_MINOR_BREAKS,
            major_breaks=FRENCH_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="fr_FR",
            babel_locale="fr_FR",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class FrenchPhonemizer(SqlitePhonemizer):
    """Phonemizer for French (Français)"""

    def __init__(
        self, lang_dir: typing.Union[str, Path], no_liason: bool = False, **kwargs
    ):
        self.lang_dir = lang_dir
        self.no_liason = no_liason

        super().__init__(
            minor_breaks=FRENCH_MINOR_BREAKS, major_breaks=FRENCH_MAJOR_BREAKS, **kwargs
        )

    def phonemize(
        self, tokens: typing.Sequence[TOKEN_OR_STR]
    ) -> typing.Iterable[WORD_PHONEMES]:
        """Add liasons to a sentence by examining word texts, parts of speech, and phonemes."""
        token_phonemes = super().phonemize(tokens)

        if self.no_liason:
            # Liasons disabled
            yield from token_phonemes

        if self.word_break:
            # Filter out word breaks
            token_phonemes = [ps for ps in token_phonemes if ps != [self.word_break]]

            # First word break
            yield [self.word_break]

        for (token1, token1_pron), (token2, token2_pron) in pairwise(
            zip(
                itertools.chain(tokens, [None]), itertools.chain(token_phonemes, [None])
            )
        ):
            if token2 is None:
                # Last token
                yield token1_pron
                continue

            liason = False

            # Conditions to meet for liason check:
            # 1) token 1 ends with a silent consonant
            # 2) token 2 starts with a vowel (phoneme)

            last_char1 = token1.text[-1]
            ends_silent_consonant = FrenchPhonemizer._has_silent_consonant(
                last_char1, token1_pron[-1]
            )
            starts_vowel = FrenchPhonemizer._is_vowel(token2_pron[0])

            token1_pos = token1.features.get(TokenFeatures.PART_OF_SPEECH)
            token2_pos = token2.features.get(TokenFeatures.PART_OF_SPEECH)

            if ends_silent_consonant and starts_vowel:
                # Handle mandatory liason cases
                # https://www.commeunefrancaise.com/blog/la-liaison

                if token1.text == "et":
                    # No liason
                    pass
                elif token1_pos in {"DET", "NUM"}:
                    # Determiner/adjective -> noun
                    liason = True
                elif (token1_pos == "PRON") and (token2_pos in {"AUX", "VERB"}):
                    # Pronoun -> verb
                    liason = True
                elif (token1_pos == "ADP") or (token1.text == "très"):
                    # Preposition
                    liason = True
                elif (token1_pos == "ADJ") and (token2_pos in {"NOUN", "PROPN"}):
                    # Adjective -> noun
                    liason = True
                elif token1_pos in {"AUX", "VERB"}:
                    # Verb -> vowel
                    liason = True

            if liason:
                # Apply liason
                # s -> z
                # p -> p
                # d|t -> d
                liason_pron = token1_pron

                if last_char1 in {"s", "x", "z"}:
                    liason_pron.append("z")
                elif last_char1 == "d":
                    liason_pron.append("t")
                elif last_char1 in {"t", "p", "n"}:
                    # Final phoneme is same as char
                    liason_pron.append(last_char1)

                yield liason_pron
            else:
                # Keep pronunciations the same
                yield token1_pron

            if self.word_break:
                # Add word breaks back in
                yield [self.word_break]

    @staticmethod
    def _has_silent_consonant(last_char: str, last_phoneme: str) -> bool:
        """True if last consonant is silent in French"""
        # Credit: https://github.com/Remiphilius/PoemesProfonds/blob/master/lecture.py

        if last_char in {"d", "p", "t"}:
            return last_phoneme != last_char
        if last_char == "r":
            return last_phoneme != "ʁ"
        if last_char in {"s", "x", "z"}:
            return last_phoneme not in {"s", "z"}
        if last_char == "n":
            return last_phoneme not in {"n", "ŋ"}

        return False

    @staticmethod
    def _is_vowel(phoneme: str) -> bool:
        """True if phoneme is a French vowel"""
        return phoneme in {
            "i",
            "y",
            "u",
            "e",
            "ø",
            "o",
            "ə",
            "ɛ",
            "œ",
            "ɔ",
            "a",
            "ɔ̃",
            "ɛ̃",
            "ɑ̃",
            "œ̃",
        }


# -----------------------------------------------------------------------------
# it-it
# -----------------------------------------------------------------------------

ITALIAN_MINOR_BREAKS = {",", ":", ";"}
ITALIAN_MAJOR_BREAKS = {".", "?", "!"}


class ItalianTokenizer(RegexTokenizer):
    """Tokenizer for Italian (Italiano)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("it_IT")
        currency_names["€"] = "EUR"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={'"', ",", ";", ":", ".", "?", "!", "„", "“", "”", "«", "»"},
            minor_breaks=ITALIAN_MINOR_BREAKS,
            major_breaks=ITALIAN_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="it_IT",
            babel_locale="it_IT",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class ItalianPhonemizer(SqlitePhonemizer):
    """Phonemizer for Italian (Italiano)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=ITALIAN_MINOR_BREAKS,
            major_breaks=ITALIAN_MAJOR_BREAKS,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# nl
# -----------------------------------------------------------------------------

DUTCH_MINOR_BREAKS = {",", ":", ";"}
DUTCH_MAJOR_BREAKS = {".", "?", "!"}


class DutchTokenizer(RegexTokenizer):
    """Tokenizer for Dutch (Nederlands)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("nl")
        currency_names["€"] = "EUR"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={'"', ",", ";", ":", ".", "?", "!", "„", "“", "”", "«", "»"},
            minor_breaks=DUTCH_MINOR_BREAKS,
            major_breaks=DUTCH_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="nl",
            babel_locale="nl",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class DutchPhonemizer(SqlitePhonemizer):
    """Phonemizer for Dutch (Nederlands)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=DUTCH_MINOR_BREAKS, major_breaks=DUTCH_MAJOR_BREAKS, **kwargs
        )


# -----------------------------------------------------------------------------
# pt
# -----------------------------------------------------------------------------

PORTUGUESE_MINOR_BREAKS = {",", ":", ";"}
PORTUGUESE_MAJOR_BREAKS = {".", "?", "!"}


class PortugueseTokenizer(RegexTokenizer):
    """Tokenizer for Portuguese (Português)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("pt")
        currency_names["€"] = "EUR"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={
                '"',
                ",",
                ";",
                ":",
                ".",
                "?",
                "¿",
                "!",
                "¡",
                "„",
                "“",
                "”",
                "«",
                "»",
            },
            minor_breaks=PORTUGUESE_MINOR_BREAKS,
            major_breaks=PORTUGUESE_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="pt",
            babel_locale="pt",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class PortuguesePhonemizer(SqlitePhonemizer):
    """Phonemizer for Portuguese (Português)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=PORTUGUESE_MINOR_BREAKS,
            major_breaks=PORTUGUESE_MAJOR_BREAKS,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# ru-ru
# -----------------------------------------------------------------------------

RUSSIAN_MINOR_BREAKS = {",", ":", ";"}
RUSSIAN_MAJOR_BREAKS = {".", "?", "!"}


class RussianTokenizer(RegexTokenizer):
    """Tokenizer for Russian (Русский)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("ru_RU")
        currency_names["₽"] = "RUB"

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={'"', ",", ";", ":", ".", "?", "!", "„", "“", "”", "«", "»"},
            minor_breaks=RUSSIAN_MINOR_BREAKS,
            major_breaks=RUSSIAN_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="ru_RU",
            babel_locale="ru_RU",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class RussianPhonemizer(SqlitePhonemizer):
    """Phonemizer for Russian (Русский)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=RUSSIAN_MINOR_BREAKS,
            major_breaks=RUSSIAN_MAJOR_BREAKS,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# sv-se
# -----------------------------------------------------------------------------

SWEDISH_MINOR_BREAKS = {",", ":", ";"}
SWEDISH_MAJOR_BREAKS = {".", "?", "!"}


class SwedishTokenizer(RegexTokenizer):
    """Tokenizer for Swedish (svenska)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("sv_SE")

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={'"', ",", ";", ":", ".", "?", "!", "„", "“", "”", "«", "»"},
            minor_breaks=SWEDISH_MINOR_BREAKS,
            major_breaks=SWEDISH_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="sv_SE",
            babel_locale="sv_SE",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class SwedishPhonemizer(SqlitePhonemizer):
    """Phonemizer for Swedish (svenska)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=SWEDISH_MINOR_BREAKS,
            major_breaks=SWEDISH_MAJOR_BREAKS,
            **kwargs,
        )


# -----------------------------------------------------------------------------
# sw
# -----------------------------------------------------------------------------

SWAHILI_MINOR_BREAKS = {",", ":", ";"}
SWAHILI_MAJOR_BREAKS = {".", "?", "!"}


class SwahiliTokenizer(RegexTokenizer):
    """Tokenizer for Swahili (Kiswahili)"""

    def __init__(
        self,
        lang_dir: typing.Union[str, Path],
        use_number_converters: bool = False,
        do_replace_currency: bool = True,
        **kwargs,
    ):
        self.lang_dir = Path(lang_dir)

        currency_names = get_currency_names("sw")

        super().__init__(
            replacements=[
                ("\\B'", '"'),  # replace single quotes
                ("'\\B", '"'),
                ('[\\<\\>\\(\\)\\[\\]"]+', ""),  # drop brackets/quotes
                ("’", "'"),  # normalize apostrophe
            ],
            punctuations={'"', ",", ";", ":", ".", "?", "!", "„", "“", "”", "«", "»"},
            minor_breaks=SWAHILI_MINOR_BREAKS,
            major_breaks=SWAHILI_MAJOR_BREAKS,
            casing_func=str.lower,
            num2words_lang="sw",
            babel_locale="sw",
            currency_names=currency_names,
            use_number_converters=use_number_converters,
            do_replace_currency=do_replace_currency,
            **kwargs,
        )


class SwahiliPhonemizer(SqlitePhonemizer):
    """Phonemizer for Swahili (svenska)"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=SWAHILI_MINOR_BREAKS,
            major_breaks=SWAHILI_MAJOR_BREAKS,
            **kwargs,
        )
