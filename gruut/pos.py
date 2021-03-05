#!/usr/bin/env python3
"""
Part of speech tagging using sklearn CRF suite.

Credit to: https://towardsdatascience.com/pos-tagging-using-crfs-ea430c5fb78b
"""
import argparse
import logging
import pickle
import string
import typing
from pathlib import Path

import sklearn_crfsuite
from sklearn_crfsuite import metrics

_LOGGER = logging.getLogger("gruut.pos")


# -----------------------------------------------------------------------------


def local_features(token, prefix=""):
    """Get features for a single word"""
    if isinstance(token, str):
        word = token.lower()
    else:
        word = token["form"].lower()

    return {
        f"{prefix}bias": 1.0,
        f"{prefix}word": word,
        f"{prefix}len(word)": len(word),
        f"{prefix}word[:4]": word[:4],
        f"{prefix}word[:3]": word[:3],
        f"{prefix}word[:2]": word[:2],
        f"{prefix}word[-3:]": word[-3:],
        f"{prefix}word[-2:]": word[-2:],
        f"{prefix}word[-4:]": word[-4:],
        f"{prefix}word.ispunctuation": (word in string.punctuation),
        f"{prefix}word.isdigit()": word.isdigit(),
    }


def word2features(sent, i):
    """Get features for a word and surrounding context"""
    token = sent[i]
    features = local_features(token)

    if i > 0:
        # One word back
        token_p1 = sent[i - 1]
        features.update(local_features(token_p1, prefix="-1:"))
    else:
        features["BOS"] = True

    if i > 1:
        # Two words back
        token_p2 = sent[i - 2]
        features.update(local_features(token_p2, prefix="-2:"))

    if i < len(sent) - 1:
        # One word forward
        token_n1 = sent[i + 1]
        features.update(local_features(token_n1, prefix="+1:"))
    else:
        features["EOS"] = True

    if i < len(sent) - 2:
        # Two words forward
        token_n2 = sent[i + 2]
        features.update(local_features(token_n2, prefix="+1:"))

    return features


def sent2features(sent):
    """Get features for all words in a sentence"""
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    """Get target labels for all words in a sentence"""
    return [token["xpos"] for token in sent]


# -----------------------------------------------------------------------------


def load_model(model_path: typing.Union[str, Path]) -> sklearn_crfsuite.CRF:
    """Load a CRF model from a pickle file"""
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
        if not isinstance(model, sklearn_crfsuite.CRF):
            _LOGGER.warning(
                "Unknown model type (expected %s, got %s)",
                sklearn_crfsuite.CRF.__name__,
                model.__class__.__name__,
            )

        return model


# -----------------------------------------------------------------------------


def train_model(
    train_path: typing.Union[str, Path],
    model_path: typing.Optional[typing.Union[str, Path]],
    max_iterations: int = 100,
) -> sklearn_crfsuite.CRF:
    """Train a new model from CONLLU data"""
    try:
        import conllu
    except ImportError as e:
        _LOGGER.fatal("conllu package is required for training")
        _LOGGER.fatal("pip install 'conllu>=4.4'")
        raise e

    _LOGGER.debug("Loading train file (%s)", train_path)
    with open(train_path, "r") as train_file:
        train_sents = conllu.parse(train_file.read())

    _LOGGER.debug("Getting features for %s train sentence(s)", len(train_sents))
    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    _LOGGER.debug("Training model for %s max iteration(s)", max_iterations)
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.25,
        c2=0.3,
        max_iterations=max_iterations,
        all_possible_transitions=True,
    )

    crf.fit(x_train, y_train)
    _LOGGER.info("Model successfully trained")

    if model_path:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        _LOGGER.debug("Writing model to %s", model_path)
        with open(model_path, "wb") as model_file:
            pickle.dump(crf, model_file)

        _LOGGER.info("Wrote model to %s", model_path)

    return crf


def do_train(
    train_path: typing.Union[str, Path],
    model_path: typing.Union[str, Path],
    max_iterations: int = 100,
    **kwargs,
):
    """CLI method for train_model"""
    train_model(train_path, model_path, max_iterations=max_iterations)


# -----------------------------------------------------------------------------


def test_model(
    model: sklearn_crfsuite.CRF,
    test_path: typing.Union[str, Path],
    out_file: typing.Optional[typing.TextIO] = None,
):
    """Print an accuracy report for a model to a file"""
    try:
        import conllu
    except ImportError as e:
        _LOGGER.fatal("conllu package is required for testing")
        _LOGGER.fatal("pip install 'conllu>=4.4'")
        raise e

    _LOGGER.debug("Loading test file (%s)", test_path)
    with open(test_path, "r") as test_file:
        test_sents = conllu.parse(test_file.read())

    _LOGGER.debug("Getting features for %s test sentence(s)", len(test_sents))
    x_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    labels = list(model.classes_)

    y_pred = model.predict(x_test)
    print(
        "F1 score on the test set = {}".format(
            metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)
        ),
        file=out_file,
    )
    print(
        "Accuracy on the test set = {}".format(
            metrics.flat_accuracy_score(y_test, y_pred)
        ),
        file=out_file,
    )

    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(
        "Test set classification report: {}".format(
            metrics.flat_classification_report(
                y_test, y_pred, labels=sorted_labels, digits=3
            )
        ),
        file=out_file,
    )


def do_test(
    model_path: typing.Union[str, Path], test_path: typing.Union[str, Path], **kwargs
):
    """CLI method for test_model"""
    _LOGGER.debug("Loading model from %s", model_path)
    model = load_model(model_path)

    test_model(model, test_path)


# -----------------------------------------------------------------------------


def predict(
    model: sklearn_crfsuite.CRF, sentences: typing.List[typing.List[str]]
) -> typing.List[typing.List[str]]:
    """Predict POS tags for sentences"""
    features = [sent2features(s) for s in sentences]
    return model.predict(features)


def do_predict(model_path: typing.Union[str, Path], texts: typing.List[str], **kwargs):
    """CLI method for predict"""
    _LOGGER.debug("Loading model from %s", model_path)
    model = load_model(model_path)

    sentences = [text.split() for text in texts]
    pos = predict(model, sentences)

    for sent, sent_pos in zip(sentences, pos):
        print(list(zip(sent, sent_pos)))


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="gruut.pos")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # -----
    # Train
    # -----
    train_parser = sub_parsers.add_parser(
        "train", help="Train a new POS model from CONLLU train/test files"
    )
    train_parser.add_argument("train_path", help="CONLLU file with training data")
    train_parser.add_argument("model_path", help="Path to write model pickle")
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
    test_parser = sub_parsers.add_parser("test", help="Test an existing POS model")
    test_parser.add_argument("model_path", help="Path to read model pickle")
    test_parser.add_argument("test_path", help="CONLLU file with testing data")
    test_parser.set_defaults(func=do_test)

    # -------
    # Predict
    # -------
    predict_parser = sub_parsers.add_parser(
        "predict", help="Predict POS tags for sentence(s)"
    )
    predict_parser.add_argument("model_path", help="Path to read model pickle")
    predict_parser.add_argument("texts", action="append", help="Sentences")
    predict_parser.set_defaults(func=do_predict)

    # ----------------
    # Shared arguments
    # ----------------
    for sub_parser in [train_parser, test_parser, predict_parser]:
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    args.func(**vars(args))

    # with open("model.pkl", "rb") as f:
    #     crf = pickle.load(f)

    # test = crf.predict_single(
    #     sent2features(["he", "wound", "the", "winding", "wind", "around", "the", "wound"])
    # )

    # print(test)

    # print(list((w["form"], w["xpos"]) for w in test_sents[3]))

    # x_train = [sent2features(s) for s in train_sents]
    # x_test = [sent2features(s) for s in test_sents]

    # y_train = [sent2labels(s) for s in train_sents]
    # y_test = [sent2labels(s) for s in test_sents]

    # # x_train, x_test, y_train, y_test = train_test_split(x, y)

    # print("Train:", len(x_train), "Test:", len(x_test))

    # crf = sklearn_crfsuite.CRF(
    #     algorithm="lbfgs",
    #     c1=0.25,
    #     c2=0.3,
    #     max_iterations=100,
    #     all_possible_transitions=True,
    # )
    # crf.fit(x_train, y_train)

    # with open("model.pkl", "wb") as f:
    #     pickle.dump(crf, f)

    # print("Done")

    # labels = list(crf.classes_)
    # labels.remove("X")

    # y_pred = crf.predict(x_train)
    # print(
    #     "F1 score on the train set = {}".format(
    #         metrics.flat_f1_score(y_train, y_pred, average="weighted", labels=labels)
    #     )
    # )
    # print(
    #     "Accuracy on the train set = {}".format(
    #         metrics.flat_accuracy_score(y_train, y_pred)
    #     )
    # )

    # sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    # print(
    #     "Train set classification report: {}".format(
    #         metrics.flat_classification_report(
    #             y_train, y_pred, labels=sorted_labels, digits=3
    #         )
    #     )
    # )

    # y_pred = crf.predict(x_test)
    # print(
    #     "F1 score on the test set = {}".format(
    #         metrics.flat_f1_score(y_test, y_pred, average="weighted", labels=labels)
    #     )
    # )
    # print(
    #     "Accuracy on the test set = {}".format(
    #         metrics.flat_accuracy_score(y_test, y_pred)
    #     )
    # )

    # sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    # print(
    #     "Test set classification report: {}".format(
    #         metrics.flat_classification_report(
    #             y_test, y_pred, labels=sorted_labels, digits=3
    #         )
    #     )
    # )
