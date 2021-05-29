"""Language-specific tokenizers and phonemizers"""
import itertools
import logging
import typing
from pathlib import Path

from .const import WORD_PHONEMES, Token, TokenFeatures, WordPronunciation
from .phonemize import Phonemizer, SqlitePhonemizer
from .toksen import RegexTokenizer, Tokenizer
from .utils import find_lang_dir, get_currency_names

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
}

ENGLISH_LANGS = {"en-us", "en-gb"}

# Languages that are expected to have a model directory
KNOWN_LANGS = set(itertools.chain(ENGLISH_LANGS, LANG_ALIASES.values()))


def resolve_lang(lang: str) -> str:
    """Try to resolve language using aliases"""
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
    """Get language-specific tokenizer"""
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

    if lang == "pt":
        assert lang_dir is not None
        return PortugueseTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "ru-ru":
        assert lang_dir is not None
        return RussianTokenizer(lang_dir=lang_dir, **kwargs)

    if lang == "sv-se":
        assert lang_dir is not None
        return SwedishTokenizer(lang_dir=lang_dir, **kwargs)

    # Fall back to basic regex tokenizer
    return RegexTokenizer(**kwargs)


def get_phonemizer(
    lang: str,
    lang_dir: typing.Optional[typing.Union[str, Path]] = None,
    no_g2p: bool = False,
    **kwargs,
) -> Phonemizer:
    """Get language-specific phonemizer"""
    lang = resolve_lang(lang)

    if (lang_dir is None) and (lang in KNOWN_LANGS):
        lang_dir = find_lang_dir(lang)

    if lang_dir is not None:
        lang_dir = Path(lang_dir)

    if lang in KNOWN_LANGS:
        assert lang_dir is not None
        if "database" not in kwargs:
            # Use database in model directory (required)
            kwargs["database"] = str(lang_dir / "lexicon.db")

        if "g2p_model" not in kwargs:
            # Use grapheme to phoneme model in model directory (optional)
            g2p_model = lang_dir / "g2p" / "model.crf"
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
        return FrenchPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "it-it":
        assert lang_dir is not None
        return ItalianPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "pt":
        assert lang_dir is not None
        return PortuguesePhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "ru-ru":
        assert lang_dir is not None
        return RussianPhonemizer(lang_dir=lang_dir, **kwargs)

    if lang == "sv-se":
        assert lang_dir is not None
        return SwedishPhonemizer(lang_dir=lang_dir, **kwargs)

    # Fall back to basic sqlite phonemizer.
    # This will fail if no database argument is provided.
    return SqlitePhonemizer(**kwargs)


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

ENGLISH_PUNCTUATIONS = {'"', ",", ";", ":", ".", "?", "!"}
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

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks=FRENCH_MINOR_BREAKS, major_breaks=FRENCH_MAJOR_BREAKS, **kwargs
        )


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
