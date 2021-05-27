"""Language-specific tokenizers and phonemizers"""
import itertools
import typing
from pathlib import Path

from .const import WORD_PHONEMES, Token, TokenFeatures, WordPronunciation
from .phonemize import Phonemizer, SqlitePhonemizer
from .toksen import RegexTokenizer, Tokenizer
from .utils import find_lang_dir, get_currency_names

# -----------------------------------------------------------------------------

ENGLISH_LANGS = {"en", "en-us", "en-gb"}

# Languages that are expected to have a model directory
KNOWN_LANGS = set(itertools.chain(ENGLISH_LANGS, {"fa"}))


def get_tokenizer(
    lang: str, lang_dir: typing.Optional[typing.Union[str, Path]] = None, **kwargs
) -> Tokenizer:
    """Get language-specific tokenizer"""
    if (lang_dir is None) and (lang in KNOWN_LANGS):
        lang_dir = find_lang_dir(lang)

    if lang in ENGLISH_LANGS:
        return EnglishTokenizer(lang_dir=lang_dir, **kwargs)
    elif lang == "fa":
        return FarsiTokenizer(lang_dir=lang_dir, **kwargs)

    # Fall back to basic regex tokenizer
    return RegexTokenizer(**kwargs)


def get_phonemizer(
    lang: str, lang_dir: typing.Union[str, Path], **kwargs
) -> Phonemizer:
    """Get language-specific phonemizer"""
    if (lang_dir is None) and (lang in KNOWN_LANGS):
        lang_dir = find_lang_dir(lang)

    if (lang in KNOWN_LANGS) and ("database" not in kwargs):
        # Use database in model directory
        kwargs["database"] = str(lang_dir / "lexicon.db")

    if lang in ENGLISH_LANGS:
        return EnglishPhonemizer(lang_dir=lang_dir, **kwargs)
    elif lang == "fa":
        return FarsiPhonemizer(lang_dir=lang_dir, **kwargs)

    # Fall back to basic sqlite phonemizer
    return SqlitePhonemizer(**kwargs)


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
                # Excluding less-used abbreviations to avoid mistakes
                "capt": "captain",
                "co": "company",
                # "col": "colonel",
                "dr": "doctor",
                "drs": "doctors",
                # "esq": "esquire",
                # "ft": "fort",
                # "gen": "general",
                # "hon": "honorable",
                "jr": "junior",
                "ltd": "limited",
                # "lt": "lieutenant",
                # "maj": "major",
                "mr": "mister",
                "mrs": "misess",
                # "rev": "reverend",
                # "sgt": "sergeant",
                # "st": "saint",
            },
            punctuations=ENGLISH_PUNCTUATIONS,
            minor_breaks=ENGLISH_MINOR_BREAKS,
            major_breaks=ENGLISH_MAJOR_BREAKS,
            drop_char_after_abbreviation=".",
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
            **kwargs,
        )


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
    """Phonemizer for Farsi/Persian"""

    def __init__(self, lang_dir: typing.Union[str, Path], **kwargs):
        self.lang_dir = lang_dir

        super().__init__(
            minor_breaks={"،", ":", ";"}, major_breaks={".", "؟", "!"}, **kwargs
        )

    def post_phonemize(
        self, token: Token, token_pron: WordPronunciation
    ) -> typing.Sequence[WORD_PHONEMES]:
        """Post-process tokens/pronunciton after phonemization (called in phonemize)"""
        phonemes = super().post_phonemize(token, token_pron)

        # Genitive case
        pos = token.features.get(TokenFeatures.PART_OF_SPEECH)
        if pos == "Ne":
            phonemes.append("e̞")

        return phonemes
