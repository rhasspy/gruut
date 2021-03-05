"""Language class for gruut"""
import logging
import os
import shutil
import typing
from pathlib import Path

import pydash
import yaml

from gruut_ipa import IPA, Phonemes

from .phonemize import Phonemizer
from .toksen import PostTokenizeFunc, Token, TokenizeFunc, Tokenizer
from .utils import env_constructor

# -----------------------------------------------------------------------------

__version__ = "0.7.0"

_LOGGER = logging.getLogger("gruut")

_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------


class Language:
    """Configuation, tokenizer, and phonemizer for a language"""

    def __init__(
        self,
        config,
        language: typing.Optional[str] = None,
        preload_lexicon: bool = False,
        custom_tokenize: typing.Optional[TokenizeFunc] = None,
        custom_post_tokenize: typing.Optional[PostTokenizeFunc] = None,
    ):
        if language is None:
            self.language = pydash.get(config, "language.code")
        else:
            self.language = language

        self.config = config

        self.tokenizer = Tokenizer(
            config,
            custom_tokenize=custom_tokenize,
            custom_post_tokenize=custom_post_tokenize,
        )

        self.phonemizer = Phonemizer(config, preload_lexicon=preload_lexicon)
        self.phonemizer.is_word = self.tokenizer.is_word  # type: ignore

        self.phonemes = Phonemes.from_language(self.language)
        self.accents: typing.Dict[str, typing.Dict[str, typing.List[str]]] = {}

        # If True, primary/seconary stress should be kept during phonemization
        self.keep_stress = bool(pydash.get(self.config, "language.keep_stress", False))

        # If True, acute/grave accents should be kept during phonemization
        self.keep_accents = bool(
            pydash.get(self.config, "language.keep_accents", False)
        )

        # Allowable tones in the language
        self.tones: typing.List[str] = pydash.get(self.config, "language.tones", [])

        # Load language-specific "accents" (different than acute/grave)
        accents = self.config.get("accents", {})
        for accent_lang, accent_map in accents.items():
            final_map = {}
            for from_phoneme, to_phonemes in accent_map.items():
                if isinstance(to_phonemes, str):
                    to_phonemes = [to_phonemes]

                final_map[from_phoneme] = to_phonemes

            self.accents[accent_lang] = final_map

    def id_to_phonemes(
        self, pad="_", no_pad=False, no_word_break=False
    ) -> typing.List[str]:
        """Return map of integer ids to phonemes"""
        # Pad symbol must always be first (index 0)
        pad = "_"

        # Acute/grave accents (' and ²)
        accents = []
        if self.keep_accents:
            accents = [IPA.ACCENT_ACUTE.value, IPA.ACCENT_GRAVE.value]

        # Primary/secondary stress (ˈ and ˌ)
        # NOTE: Accute accent (0x0027) != primary stress (0x02C8)
        stresses = []
        if self.keep_stress:
            stresses = [IPA.STRESS_PRIMARY.value, IPA.STRESS_SECONDARY.value]

        # Tones
        tones = self.tones

        # Word break
        word_break = [IPA.BREAK_WORD.value]
        if no_word_break:
            word_break = []

        phonemes_list = [pad]
        if no_pad:
            phonemes_list = []

        # Always include pad and break symbols.
        # In the future, intontation/tones should also be added.
        phonemes_list = (
            phonemes_list
            + [IPA.BREAK_MINOR.value, IPA.BREAK_MAJOR.value]
            + word_break
            + accents
            + stresses
            + tones
            + sorted([p.text for p in self.phonemes])
        )

        return phonemes_list

    # -------------------------------------------------------------------------

    @staticmethod
    def load(
        lang_dir: Path,
        language: str,
        preload_lexicon: bool = False,
        custom_tokenizers: bool = True,
    ) -> typing.Optional["Language"]:
        """Load language from code"""

        # Expand environment variables in string value
        yaml.SafeLoader.add_constructor("!env", env_constructor)

        # Load configuration
        config_path = lang_dir / "language.yml"

        if not config_path.is_file():
            _LOGGER.warning("Missing %s", config_path)
            return None

        # Set environment variable for config loading
        os.environ["config_dir"] = str(config_path.parent)
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Language-specific loading
        custom_tokenize: typing.Optional[TokenizeFunc] = None
        custom_post_tokenize: typing.Optional[PostTokenizeFunc] = None

        if custom_tokenizers:
            if language == "fa":
                # Use hazm for text normalization and POS tagging.
                custom_tokenize = Language.make_fa_tokenize(lang_dir)
            elif language in ("en-us", "en-gb") and shutil.which("java"):
                # Use the Stanford POS tagger.
                # Requires java, so don't bother if it's not available.
                custom_post_tokenize = Language.make_en_post_tokenize(lang_dir)

        return Language(
            config=config,
            language=language,
            preload_lexicon=preload_lexicon,
            custom_tokenize=custom_tokenize,
            custom_post_tokenize=custom_post_tokenize,
        )

    @staticmethod
    def make_en_post_tokenize(lang_dir: Path) -> typing.Optional[PostTokenizeFunc]:
        """Tokenization post-processing for English"""
        from .pos import load_model, predict

        # Load part of speech tagger
        pos_dir = lang_dir / "pos"
        model_path = pos_dir / "model.pkl"

        if not (model_path.is_file()):
            _LOGGER.warning("Missing POS model: %s", model_path)
            return None

        _LOGGER.debug("Loading POS model from %s", model_path)
        pos_model = load_model(model_path)

        pos_map = {
            "NNS": "NN",
            "NNP": "NN",
            "NNPS": "NN",
            "PRP$": "PRP",
            "RBR": "RB",
            "RBS": "RB",
            "VBG": "VB",
            "VBN": "VBD",
            "VBP": "VB",
            "VBZ": "VB",
            "JJR": "JJ",
            "JJS": "JJ",
        }

        def do_post_tokenize(
            sentence_tokens: typing.List[Token], **kwargs
        ) -> typing.List[Token]:
            """Tag part of speech for sentence tokens"""
            guess_pos = kwargs.get("guess_pos", True)
            if not guess_pos:
                # Don't run tagger is POS isn't needed
                return sentence_tokens

            words = [t.text for t in sentence_tokens]
            sents = [words]

            sents_pos = predict(pos_model, sents)
            assert sents_pos, "No POS predictions"
            words_pos = sents_pos[0]
            assert len(words_pos) == len(words), f"Length mismatch for words/pos"

            for i, pos in enumerate(words_pos):
                sentence_tokens[i].pos = pos_map.get(pos, pos)

            return sentence_tokens

        return do_post_tokenize

    @staticmethod
    def make_fa_tokenize(lang_dir: Path) -> typing.Optional[TokenizeFunc]:
        """Tokenize Persian/Farsi"""
        try:
            import hazm
        except ImportError:
            _LOGGER.warning("hazm is highly recommended for language 'fa'")
            _LOGGER.warning("pip install 'hazm>=0.7.0'")
            return None

        normalizer = hazm.Normalizer()

        # Load part of speech tagger
        model_path = lang_dir / "postagger.model"

        if not model_path.is_file():
            _LOGGER.warning("Missing model: %s", model_path)
            return None

        _LOGGER.debug("Using hazm tokenizer (model=%s)", model_path)
        tagger = hazm.POSTagger(model=str(model_path))

        def do_tokenize(text: str, **kwargs) -> typing.List[typing.List[Token]]:
            """Normalize, tokenize, and recognize part of speech"""
            sentences_tokens = []
            sentences = hazm.sent_tokenize(normalizer.normalize(text))
            for sentence in sentences:
                sentence_tokens = []
                for word, pos in tagger.tag(hazm.word_tokenize(sentence)):
                    sentence_tokens.append(Token(text=word, pos=pos))

                sentences_tokens.append(sentence_tokens)

            return sentences_tokens

        return do_tokenize
