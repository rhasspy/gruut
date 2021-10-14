#!/usr/bin/env python3
"""
Part of speech tagging using python CRF suite.

Credit to: https://towardsdatascience.com/pos-tagging-using-crfs-ea430c5fb78b

Training requires conllu package:

.. code-block:: sh

    pip install conllu

Training data comes from Univeral Dependencies (https://universaldependencies.org/)

Example:

.. code-block:: sh

    python3 -m gruut.pos train --conllu train.conllu --output model.crf --label xpos

Pre-trained models have the following settings:

* c1 = 0.25
* c2 = 0.3
* max-iterations = 100

English model is trained with "xpos" label.
French model is trained with "upos" label.
"""
import argparse
import base64
import logging
import os
import string
import sys
import time
import typing
from pathlib import Path

import jsonlines
import pycrfsuite

_LOGGER = logging.getLogger("gruut.pos")

# -----------------------------------------------------------------------------

FEATURES_TYPE = typing.Dict[
    str, typing.Union[str, bool, int, float, typing.Sequence[str]]
]


class PartOfSpeechTagger:
    """Part of speech tagger using a pre-trained CRF model"""

    def __init__(
        self, crf_tagger: typing.Union[str, Path, pycrfsuite.Tagger], **kwargs
    ):
        if isinstance(crf_tagger, pycrfsuite.Tagger):
            self.crf_tagger = crf_tagger
        else:
            # Load model
            self.crf_tagger = pycrfsuite.Tagger()
            self.crf_tagger.open(str(crf_tagger))

    def __call__(self, words: typing.Sequence[str]) -> typing.Sequence[str]:
        """Returns POS tag for each word"""
        features = PartOfSpeechTagger.sent2features(words)
        return self.crf_tagger.tag(features)

    @staticmethod
    def local_features(
        word: str,
        prefix: str = "",
        bias: float = 1.0,
        add_punctuation: bool = True,
        add_digit: bool = True,
        add_length: bool = True,
        chars_front: int = 3,
        chars_back: int = 3,
        encode: bool = True,
    ) -> FEATURES_TYPE:
        """Get features for a single word"""
        features: FEATURES_TYPE = {
            f"{prefix}bias": bias,
            f"{prefix}word": PartOfSpeechTagger.encode_string(word) if encode else word,
        }

        if add_length:
            features[f"{prefix}len(word)"] = len(word)

        if add_punctuation:
            features[f"{prefix}word.ispunctuation"] = word in string.punctuation

        if add_digit:
            features[f"{prefix}word.isdigit()"] = word.isdigit()

        # Chunks from front
        for i in range(2, chars_front + 1):
            features[f"{prefix}word[:{i}]"] = word[:i]

        # Chunks from pack
        for i in range(2, chars_back + 1):
            features[f"{prefix}word[-{i}:]"] = word[-i:]

        return features

    @staticmethod
    def word2features(
        sentence: typing.Sequence[str],
        i: int,
        add_bos: bool = True,
        add_eos: bool = True,
        words_backward: int = 2,
        words_forward: int = 2,
        **kwargs,
    ) -> FEATURES_TYPE:
        """Get features for a word and surrounding context"""
        word = sentence[i]
        num_words = len(sentence)
        features = PartOfSpeechTagger.local_features(word, **kwargs)

        if (i == 0) and add_bos:
            features["BOS"] = True

        if (i == (num_words - 1)) and add_eos:
            features["EOS"] = True

        for j in range(1, words_backward + 1):
            if i >= j:
                word_prev = sentence[i - j]
                features.update(
                    PartOfSpeechTagger.local_features(
                        word_prev, prefix=f"-{j}:", **kwargs
                    )
                )

        for j in range(1, words_forward + 1):
            if i < (num_words - j):
                word_next = sentence[i + j]
                features.update(
                    PartOfSpeechTagger.local_features(
                        word_next, prefix=f"+{j}:", **kwargs
                    )
                )

        return features

    @staticmethod
    def sent2features(
        sentence: typing.Sequence[str], **kwargs
    ) -> typing.List[FEATURES_TYPE]:
        """Get features for all words in a sentence"""
        return [
            PartOfSpeechTagger.word2features(sentence, i, **kwargs)
            for i in range(len(sentence))
        ]

    @staticmethod
    def encode_string(s: str) -> str:
        """Encodes string in a form that crfsuite will accept (ASCII) and can be decoded"""
        return base64.b64encode(s.encode()).decode("ascii")

    @staticmethod
    def decode_string(s: str) -> str:
        """Decodes a string encoded by encode_string"""
        return base64.b64decode(s.encode("ascii")).decode()


# -----------------------------------------------------------------------------


def train_model(
    conllu_path: typing.Union[str, Path],
    output_path: typing.Union[str, Path],
    label: str = "xpos",
    c1: float = 0.25,
    c2: float = 0.3,
    max_iterations: int = 100,
):
    """Train a new model from CONLLU data"""
    try:
        import conllu
    except ImportError as e:
        _LOGGER.critical("conllu package is required for training")
        _LOGGER.critical("pip install 'conllu>=4.4'")
        raise e

    conllu_path = Path(conllu_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _LOGGER.debug("Loading train file (%s)", conllu_path)
    with open(conllu_path, "r", encoding="utf-8") as conllu_file:
        train_sents = conllu.parse(conllu_file.read())

    _LOGGER.debug("Training model for %s max iteration(s)", max_iterations)
    trainer = pycrfsuite.Trainer(verbose=False)

    _LOGGER.debug("Getting features for %s training sentence(s)", len(train_sents))
    for sent in train_sents:
        words = [token["form"] for token in sent]
        features = PartOfSpeechTagger.sent2features(words)

        labels = []
        skip_sent = False
        for token in sent:
            token_label = token.get(label)
            if token_label is None:
                _LOGGER.warning("Example has empty label for %s: %s", token, sent)
                skip_sent = True
                break

            labels.append(token_label)

        if skip_sent:
            continue

        trainer.append(features, labels)

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
    """CLI method for train_model"""
    train_model(
        conllu_path=args.conllu,
        output_path=args.output,
        label=args.label,
        c1=args.c1,
        c2=args.c2,
        max_iterations=args.max_iterations,
    )


# -----------------------------------------------------------------------------


def do_print_labels(args):
    """Print label set from a CONLLU file"""
    try:
        import conllu
    except ImportError as e:
        _LOGGER.critical("conllu package is required for training")
        _LOGGER.critical("pip install 'conllu>=4.4'")
        raise e

    labels = set()
    with open(args.conllu, "r", encoding="utf-8") as conllu_file:
        for sent in conllu.parse(conllu_file.read()):
            for token in sent:
                token_label = token.get(args.label)
                if token_label is not None:
                    labels.add(token_label)

    print(sorted(labels))


# -----------------------------------------------------------------------------


def do_predict(args):
    """CLI method for predict"""
    tagger = PartOfSpeechTagger(args.model)

    if args.texts:
        lines = args.texts
    else:
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading sentences from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in lines:
        line = line.strip()
        if not line:
            continue

        words = line.split()
        words_and_tags = list(zip(words, tagger(words)))

        writer.write(words_and_tags)


def do_test(args):
    """CLI method for testing"""
    try:
        import conllu
    except ImportError as e:
        _LOGGER.critical("conllu package is required for training")
        _LOGGER.critical("pip install 'conllu>=4.4'")
        raise e

    tagger = PartOfSpeechTagger(args.model)

    _LOGGER.debug("Testing file (%s)", args.conllu)

    num_sentences = 0
    num_words = 0
    sents_with_errors = 0
    total_errors = 0
    with open(args.conllu, "r", encoding="utf-8") as conllu_file:
        for sent in conllu.parse(conllu_file.read()):
            words = [token["form"] for token in sent]
            actual_labels = [token.get(args.label) for token in sent]
            expected_labels = tagger(words)

            had_error = False
            for actual, expected in zip(actual_labels, expected_labels):
                if actual != expected:
                    total_errors += 1
                    had_error = True

                num_words += 1

            if had_error:
                sents_with_errors += 1

            num_sentences += 1

    if (num_sentences < 1) or (num_words < 1):
        return

    print(
        "{0} out of {1} word(s) had an incorrect tag ({2:0.2f}%)".format(
            total_errors, num_words, total_errors / num_words
        )
    )
    print(
        "{0} out of {1} sentence(s) had at least one error ({2:0.2f}%)".format(
            sents_with_errors, num_sentences, sents_with_errors / num_sentences
        )
    )


# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="pos.py")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # -----
    # Train
    # -----
    train_parser = sub_parsers.add_parser(
        "train", help="Train a new POS model from CONLLU file"
    )
    train_parser.add_argument(
        "--conllu", required=True, help="CONLLU file with training data"
    )
    train_parser.add_argument(
        "--output", required=True, help="Path to write output model"
    )
    train_parser.add_argument(
        "--label", default="xpos", help="Field to predict in training data"
    )
    train_parser.add_argument("--c1", type=float, default=0.25, help="L1 penalty")
    train_parser.add_argument("--c2", type=float, default=0.3, help="L2 penalty")
    train_parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of iterations to train for",
    )
    train_parser.set_defaults(func=do_train)

    # ----
    # Test
    # ----
    test_parser = sub_parsers.add_parser(
        "test", help="Test a POS model on a CONLLU file"
    )
    test_parser.add_argument("--model", required=True, help="Path to POS tagger model")
    test_parser.add_argument(
        "--conllu", required=True, help="CONLLU file with testing data"
    )
    test_parser.add_argument(
        "--label", default="xpos", help="Field to predict in training data"
    )
    test_parser.set_defaults(func=do_test)

    # ------------
    # Print Labels
    # ------------
    print_labels_parser = sub_parsers.add_parser(
        "print-labels", help="Print set of unique labels from a CONLLU file"
    )
    print_labels_parser.add_argument(
        "--conllu", required=True, help="CONLLU file with training data"
    )
    print_labels_parser.add_argument(
        "--label", default="xpos", help="Field to predict in training data"
    )
    print_labels_parser.set_defaults(func=do_print_labels)

    # -------
    # Predict
    # -------
    predict_parser = sub_parsers.add_parser(
        "predict", help="Predict POS tags for sentence(s)"
    )
    predict_parser.add_argument(
        "--model", required=True, help="Path to POS tagger model"
    )
    predict_parser.add_argument("texts", nargs="*", help="Sentences")
    predict_parser.set_defaults(func=do_predict)

    # ----------------
    # Shared arguments
    # ----------------
    for sub_parser in [train_parser, predict_parser, test_parser, print_labels_parser]:
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


if __name__ == "__main__":
    main()
