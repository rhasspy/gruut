#!/usr/bin/env python3
"""
Grapheme to phoneme prediction using python CRF suite.

Training requires pre-aligned corpus in Phonetisaurus format.
https://github.com/AdolfVonKleist/Phonetisaurus

The format of this corpus is:

    t}t e}ˈɛ s}s t}t

Each line contains a single word, with graphemes and phonemes separated by "}".
Multiple graphemes are separated by "|":

    s|h}ʃ o|w}ˈoʊ

The empty phoneme is "_":

    w}w h}_ y}ˈaɪ

Example:

.. code-block:: sh

    python3 -m gruut.g2p train --corpus g2p.corpus --output model.crf

Pre-trained models have the following settings:

* c1 = 0
* c2 = 1
* max-iterations = 100
"""
import argparse
import base64
import itertools
import logging
import os
import sys
import time
import typing
import unicodedata
from pathlib import Path

import pycrfsuite

_LOGGER = logging.getLogger("gruut.g2p")

# -----------------------------------------------------------------------------

FEATURES_TYPE = typing.Dict[str, typing.Union[str, bool, int, float]]
EPS_PHONEME = "_"
PHONEME_JOIN = "|"


class GraphemesToPhonemes:
    """Grapheme to phoneme CRF tagger"""

    def __init__(
        self,
        crf_tagger: typing.Union[str, Path, pycrfsuite.Tagger],
        eps_phoneme: str = EPS_PHONEME,
        phoneme_join: str = PHONEME_JOIN,
    ):
        if isinstance(crf_tagger, pycrfsuite.Tagger):
            self.crf_tagger = crf_tagger
        else:
            # Load model
            self.crf_tagger = pycrfsuite.Tagger()
            self.crf_tagger.open(str(crf_tagger))

        # Empty phoneme (dropped)
        self.eps_phoneme = eps_phoneme

        # String used to join multiple predicted phonemes
        self.phoneme_join = phoneme_join

    def __call__(self, word: str, normalize: bool = True) -> typing.Sequence[str]:
        """Guess phonemes for word"""
        features = GraphemesToPhonemes.word2features(word, normalize=normalize)
        coded_phonemes = self.crf_tagger.tag(features)
        phonemes: typing.List[str] = []

        for coded_ps in coded_phonemes:
            decoded_ps = GraphemesToPhonemes.decode_string(coded_ps)
            for p in decoded_ps.split(self.phoneme_join):
                if p != self.eps_phoneme:
                    phonemes.append(p)

        return phonemes

    # -------------------------------------------------------------------------

    @staticmethod
    def word2features(
        word: typing.Union[str, typing.List[str]], normalize: bool = True, **kwargs
    ):
        """Create feature dicts for all graphemes in a word"""
        if normalize and isinstance(word, str):
            # Combine characters
            # See: https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
            word = unicodedata.normalize("NFC", word)

        return [
            GraphemesToPhonemes.grapheme2features(word, i, **kwargs)
            for i in range(len(word))
        ]

    @staticmethod
    def grapheme2features(
        word: typing.Union[str, typing.Sequence[str]],
        i: int,
        add_begin: bool = True,
        add_end: bool = True,
        chars_backward: int = 3,
        chars_forward: int = 3,
        bias: float = 1.0,
        encode: bool = True,
    ) -> FEATURES_TYPE:
        """Create feature dict for single grapheme"""
        g = word[i]
        num_g = len(word)

        features: FEATURES_TYPE = {
            "bias": bias,
            "grapheme": GraphemesToPhonemes.encode_string(g) if encode else g,
        }

        if (i == 0) and add_begin:
            features["begin"] = True

        for j in range(1, chars_backward + 1):
            if i >= j:
                g_prev = word[i - j]
                features[f"grapheme-{j}"] = (
                    GraphemesToPhonemes.encode_string(g_prev) if encode else g_prev
                )

        for j in range(1, chars_forward + 1):
            if i < (num_g - j):
                g_next = word[i + j]
                features[f"grapheme+{j}"] = (
                    GraphemesToPhonemes.encode_string(g_next) if encode else g_next
                )

        if (i == (num_g - 1)) and add_end:
            features["end"] = True

        return features

    @staticmethod
    def encode_string(s: str) -> str:
        """Encodes string in a form that crfsuite will accept (ASCII) and can be decoded"""
        return base64.b64encode(s.encode()).decode("ascii")

    @staticmethod
    def decode_string(s: str) -> str:
        """Decodes a string encoded by encode_string"""
        return base64.b64decode(s.encode("ascii")).decode()


# -----------------------------------------------------------------------------


def train(
    corpus_path: typing.Union[str, Path],
    output_path: typing.Union[str, Path],
    group_separator: str = "}",
    item_separator: str = "|",
    phoneme_join: str = PHONEME_JOIN,
    eps_phoneme: str = EPS_PHONEME,
    remove_phonemes: typing.Optional[typing.Iterable[str]] = None,
    c1: float = 0.0,
    c2: float = 1.0,
    max_iterations: int = 100,
):
    """Train a new G2P model"""
    corpus_path = Path(corpus_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remove_phonemes = set(remove_phonemes or [])

    trainer = pycrfsuite.Trainer(verbose=False)

    with open(corpus_path, "r") as corpus:
        for i, line in enumerate(corpus):
            line = line.strip()
            if not line:
                continue

            # Parse into graphemes and phonemes
            skip_line = False
            parts = line.split()
            aligned_g = []
            aligned_p = []

            for part in parts:
                # Graphemes/phonemes are separated by }
                gs_str, ps_str = part.split(group_separator, maxsplit=1)

                # Multiple graphemes and phonemes are separated by |
                gs = gs_str.split(item_separator)
                ps = [
                    p for p in ps_str.split(item_separator) if p not in remove_phonemes
                ]

                # Align graphemes and phonemes, allowing for empty phonemes only
                for g1, p1 in itertools.zip_longest(gs, [ps], fillvalue=None):
                    if g1 is None:
                        skip_line = True
                        break

                    aligned_g.append(g1)

                    if p1:
                        aligned_p.append(phoneme_join.join(p1))
                    else:
                        aligned_p.append(eps_phoneme)

                if skip_line:
                    break

            if skip_line:
                _LOGGER.warning(
                    "Failed to align line %s: %s (graphemes=%s, phonemes=%s)",
                    i + 1,
                    line,
                    gs,
                    ps,
                )
                continue

            # Add example to trainer
            try:
                encoded_p = [GraphemesToPhonemes.encode_string(p) for p in aligned_p]
                trainer.append(GraphemesToPhonemes.word2features(aligned_g), encoded_p)
            except Exception as e:
                _LOGGER.exception("graphemes=%s phonemes=%s", aligned_g, aligned_p)
                raise e

    trainer.set_params(
        {
            "c1": c1,  # coefficient for L1 penalty
            "c2": c2,  # coefficient for L2 penalty
            "max_iterations": max_iterations,  # stop earlier
            # include transitions that are possible, but not observed
            "feature.possible_transitions": True,
        }
    )

    _LOGGER.debug(trainer.get_params())

    # Begin training
    _LOGGER.info("Training")

    start_time = time.perf_counter()
    trainer.train(str(output_path))
    end_time = time.perf_counter()

    _LOGGER.info("Training completed in %s second(s)", end_time - start_time)
    _LOGGER.info(trainer.logparser.last_iteration)


def do_train(args):
    """CLI method for train"""
    train(
        corpus_path=args.corpus,
        output_path=args.output,
        group_separator=args.group_separator,
        item_separator=args.item_separator,
        remove_phonemes=args.remove_phonemes,
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
    )


def do_predict(args):
    """CLI method for predict"""
    tagger = GraphemesToPhonemes(args.model)

    if args.texts:
        lines = args.texts
    else:
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading words from stdin...", file=sys.stderr)

    for line in lines:
        line = line.strip()
        if not line:
            continue

        word = line
        phonemes = tagger(word)

        print(word, *phonemes)


def do_test(args):
    """CLI method for test"""
    try:
        from rapidfuzz.string_metric import levenshtein
    except ImportError as e:
        _LOGGER.fatal("rapidfuzz library is needed for levenshtein distance")
        _LOGGER.fatal("pip install 'rapidfuzz>=1.4.1'")
        raise e

    tagger = GraphemesToPhonemes(args.model)

    if args.texts:
        lines = args.texts
    else:
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading lexicon lines from stdin...", file=sys.stderr)

    num_errors = 0
    num_missing = 0
    num_phonemes = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        word, actual_phonemes = line.split(maxsplit=1)
        expected_phonemes = "".join(tagger(word))

        if expected_phonemes:
            distance = levenshtein(expected_phonemes, actual_phonemes)
            num_errors += distance
            num_phonemes += len(actual_phonemes)
        else:
            num_missing += 1
            _LOGGER.warning("No pronunciation for %s", word)

    # Calculate results
    per = round(num_errors / num_phonemes, 2)
    print("PER:", per, "Errors:", num_errors)

    if num_missing > 0:
        print("Total missing:", num_missing)


# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="g2p.py")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # -----
    # Train
    # -----
    train_parser = sub_parsers.add_parser(
        "train", help="Train a new G2P model from a pre-aligned Phonetisaurus corpus"
    )
    train_parser.add_argument(
        "--corpus", required=True, help="Path to aligned Phonetisaurus g2p corpus"
    )
    train_parser.add_argument(
        "--output", required=True, help="Path to output tagger model"
    )
    train_parser.add_argument("--c1", type=float, default=0.0, help="L1 penalty")
    train_parser.add_argument("--c2", type=float, default=1.0, help="L2 penalty")
    train_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of training iterations (default: 100)",
    )
    train_parser.add_argument(
        "--group-separator",
        default="}",
        help="Separator between graphemes and phonemes",
    )
    train_parser.add_argument(
        "--item-separator", default="|", help="Separator between items in a group"
    )
    train_parser.add_argument(
        "--remove-phonemes", nargs="*", help="Remove phonemes from examples"
    )
    train_parser.set_defaults(func=do_train)

    # -------
    # Predict
    # -------
    predict_parser = sub_parsers.add_parser(
        "predict", help="Predict phonemes for word(s)"
    )
    predict_parser.add_argument(
        "--model", required=True, help="Path to G2P tagger model"
    )
    predict_parser.add_argument("texts", nargs="*", help="Words")
    predict_parser.set_defaults(func=do_predict)

    # ----
    # Test
    # ----
    test_parser = sub_parsers.add_parser("test", help="Test G2P model on a lexicon")
    test_parser.add_argument("--model", required=True, help="Path to G2P tagger model")
    test_parser.add_argument(
        "texts", nargs="*", help="Lines with '<word> <phoneme> <phoneme> ...'"
    )
    test_parser.set_defaults(func=do_test)

    # ----------------
    # Shared arguments
    # ----------------
    for sub_parser in [train_parser, predict_parser, test_parser]:
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    args.func(args)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
