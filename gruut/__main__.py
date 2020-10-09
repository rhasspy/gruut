#!/usr/bin/env python3
"""Command-line interface to gruut"""
import argparse
import dataclasses
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import jsonlines
import pydash
import yaml

import gruut_ipa

from .utils import env_constructor, load_lexicon, pairwise

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut")

_DIR = Path(__file__).parent
_DATA_DIR = _DIR / "data"

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    # Expand environment variables in string value
    yaml.SafeLoader.add_constructor("!env", env_constructor)

    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    lang_dir = _DATA_DIR / args.language
    assert lang_dir.is_dir(), "Unsupported language"

    # Load configuration
    config_path = lang_dir / "language.yml"
    assert config_path.is_file(), f"Missing {config_path}"

    # Set environment variable for config loading
    os.environ["config_dir"] = str(config_path.parent)
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)

    args.func(config, args)


# -----------------------------------------------------------------------------


def do_tokenize(config, args):
    """
    Split lines from stdin into sentences, tokenize and clean.

    Prints a line of JSON for each sentence.
    """
    from .toksen import Tokenizer

    tokenizer = Tokenizer(config)

    if os.isatty(sys.stdin.fileno()):
        print("Reading text from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        sentences = list(
            tokenizer.tokenize(
                line,
                number_converters=args.number_converters,
                replace_currency=(not args.disable_currency),
            )
        )
        raw_words = []
        clean_words = []

        for sentence in sentences:
            raw_words.extend(sentence.raw_words)
            clean_words.extend(sentence.clean_words)

        if args.exclude_non_words:
            # Exclude punctuations, etc.
            clean_words = [w for w in clean_words if tokenizer.is_word(w)]

        writer.write(
            {
                "raw_text": line,
                "raw_words": raw_words,
                "clean_words": clean_words,
                "clean_text": " ".join(clean_words),
                "sentences": [dataclasses.asdict(s) for s in sentences],
            }
        )


# -----------------------------------------------------------------------------


def do_phonemize(config, args):
    """
    Reads JSONL from stdin with "clean_words" property.

    Looks up or guesses phonetic pronuncation(s) for all clean words.

    Prints a line of JSON for each input line.
    """
    from .phonemize import Phonemizer

    phonemizer = Phonemizer(config)

    if os.isatty(sys.stdin.fileno()):
        print("Reading tokenize JSONL from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        sentence_obj = json.loads(line)
        clean_words = sentence_obj["clean_words"]

        sentence_prons = phonemizer.phonemize(
            clean_words, word_indexes=args.word_indexes, word_breaks=args.word_breaks
        )
        sentence_obj["pronunciations"] = sentence_prons

        # Pick first pronunciation for each word
        first_pron = []
        for word_prons in sentence_prons:
            if word_prons:
                first_pron.append(word_prons[0])

        sentence_obj["pronunciation"] = first_pron

        # Create string of first pronunciation
        sentence_obj["pronunciation_text"] = args.word_separator.join(
            args.phoneme_separator.join(word_pron) for word_pron in first_pron
        )

        # Get Sampa pronunciation
        sentence_obj["sampa"] = [
            [gruut_ipa.ipa_to_sampa(phoneme) for phoneme in word_pron]
            for word_pron in first_pron
        ]

        sentence_obj["sampa_text"] = " ".join(
            "".join(word_pron) for word_pron in sentence_obj["sampa"]
        ).strip()

        # Get eSpeak pronunciation
        sentence_obj["espeak"] = [
            [gruut_ipa.ipa_to_espeak(phoneme) for phoneme in word_pron]
            for word_pron in first_pron
        ]

        sentence_obj["espeak_text"] = (
            "[["
            + " ".join(
                "".join(word_pron) for word_pron in sentence_obj["espeak"]
            ).strip()
            + "]]"
        )

        # Print back out with extra info
        writer.write(sentence_obj)


# -----------------------------------------------------------------------------


def do_phones_to_phonemes(config, args):
    """Transform/group phones in a pronuncation into language phonemes"""
    phonemes_path = Path(pydash.get(config, "language.phonemes"))

    with open(phonemes_path, "r") as phonemes_file:
        phonemes = gruut_ipa.Phonemes.from_text(phonemes_file)

    if args.phones:
        phones = args.phones
    else:
        # Read from stdin
        phones = sys.stdin
        if os.isatty(sys.stdin.fileno()):
            print("Reading pronunciations from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in phones:
        line = line.strip()
        if line:
            line_phonemes = phonemes.split(line, keep_stress=args.keep_stress)
            phonemes_list = [p.text for p in line_phonemes]

            writer.write(
                {
                    "language": args.language,
                    "raw_text": line,
                    "phonemes_text": " ".join(phonemes_list),
                    "phonemes_list": phonemes_list,
                    "phonemes": [p.to_dict() for p in line_phonemes],
                }
            )


# -----------------------------------------------------------------------------


def do_coverage(config, args):
    """Get phoneme coverage"""
    from . import Language

    gruut_lang = Language.load(args.language)

    # List of possible phonemes in the language
    phonemes = [p.text for p in gruut_lang.phonemes]

    _LOGGER.debug("Getting phoneme pairs from lexicon")

    # Get set of phoneme pairs from the lexicon.
    # This is done instead of using all possible phoneme pairs, because there
    # are many pairs that either humans cannot produce or are not used.
    # We assume the lexicon will contain an example of each useful pairs.
    all_pairs = set()
    for word_prons in gruut_lang.phonemizer.lexicon.values():
        for word_pron in word_prons:
            all_pairs.update(pairwise(word_pron))

    single_counts = Counter()
    pair_counts = Counter()

    # Process output from phonemize command
    if os.isatty(sys.stdin.fileno()):
        print("Reading phonemize JSONL from stdin...", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        # JSON object with "pronunciation" property
        phonemize_obj = json.loads(line)
        sentence_prons = []
        for word_pron in phonemize_obj.get("pronunciation", []):
            word_pron = [p for p in word_pron if not gruut_ipa.IPA.is_break(p)]
            sentence_prons.extend(word_pron)

        # Count single phonemes
        for p in sentence_prons:
            single_counts[p] += 1

        # Count phoneme pairs
        for p1, p2 in pairwise(sentence_prons):
            pair_counts[(p1, p2)] += 1

    # Print coverage report
    writer = jsonlines.Writer(sys.stdout, flush=True)
    writer.write(
        {
            "singles": {p: single_counts[p] for p in phonemes},
            "pairs": {" ".join(pair): pair_counts[pair] for pair in all_pairs},
            "coverage": {
                "single": len(single_counts) / len(phonemes),
                "pair": len(pair_counts) / len(all_pairs),
            },
        }
    )


# -----------------------------------------------------------------------------


def do_phonemize_lexicon(config, args):
    """Convert phonetic lexicon to phonemic lexicon"""
    phonemes_path = Path(pydash.get(config, "language.phonemes"))

    with open(phonemes_path, "r") as phonemes_file:
        phonemes = gruut_ipa.Phonemes.from_text(phonemes_file)

    if os.isatty(sys.stdin.fileno()):
        print("Reading lexicon from stdin...")

    lexicon = load_lexicon(sys.stdin)
    unknown_counts = Counter()

    for word, word_prons in lexicon.items():
        for word_pron in word_prons:
            word_pron_str = "".join(word_pron)
            pron_phonemes = phonemes.split(word_pron_str, keep_stress=args.keep_stress)
            pron_phonemes_str = " ".join(p.text for p in pron_phonemes).strip()

            if not pron_phonemes_str:
                # Don't print words with empty phonemic pronunciations
                _LOGGER.warning("No pronunciation for '%s': %s", word, word_pron)
                continue

            # Drop words with unknown phonemes
            unknown = []
            for phoneme in pron_phonemes:
                if phoneme.unknown:
                    unknown_counts[phoneme.text] += 1
                    unknown.append(phoneme.text)

            if unknown:
                _LOGGER.warning("Unknown phonemes in '%s': %s", word, unknown)
                continue

            print(word, pron_phonemes_str)

    if unknown_counts:
        _LOGGER.warning("%s unknown phonemes:", len(unknown_counts))
        _LOGGER.warning(unknown_counts.most_common())


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="gruut")
    parser.add_argument("language", help="Language code (e.g., en-us)")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # --------
    # tokenize
    # --------
    tokenize_parser = sub_parsers.add_parser(
        "tokenize", help="Sentencize/tokenize raw text, clean, and expand numbers"
    )
    tokenize_parser.add_argument(
        "--disable-currency",
        action="store_true",
        help="Disable automatic replacement of currency with words (e.g., $1 -> one dollar)",
    )
    tokenize_parser.add_argument(
        "--number-converters",
        action="store_true",
        help="Allow number_conv form for specifying num2words converter (cardinal, ordinal, ordinal_num, year, currency)",
    )
    tokenize_parser.add_argument(
        "--exclude-non-words",
        action="store_true",
        help="Remove punctionation, etc. from clean words",
    )
    tokenize_parser.set_defaults(func=do_tokenize)

    # ---------
    # phonemize
    # ---------
    phonemize_parser = sub_parsers.add_parser(
        "phonemize", help="Look up or guess word pronunciations from JSONL sentences"
    )
    phonemize_parser.set_defaults(func=do_phonemize)
    phonemize_parser.add_argument(
        "--word-separator",
        default=" ",
        help="Separator to add between words in output pronunciation (default: space)",
    )
    phonemize_parser.add_argument(
        "--phoneme-separator",
        default=" ",
        help="Separator to add between words in output pronunciation (default: space)",
    )
    phonemize_parser.add_argument(
        "--word-indexes",
        action="store_true",
        help="Allow word_n form for specifying nth pronunciation of word from lexicon",
    )
    phonemize_parser.add_argument(
        "--word-breaks",
        action="store_true",
        help="Add the IPA word break symbol (#) between each word",
    )

    # ---------------
    # phones2phonemes
    # ---------------
    phones2phonemes_parser = sub_parsers.add_parser(
        "phones2phonemes", help="Group phonetic pronunciation into language phonemes"
    )
    phones2phonemes_parser.set_defaults(func=do_phones_to_phonemes)
    phones2phonemes_parser.add_argument(
        "phones", nargs="*", help="Phone strings to group (default: stdin)"
    )
    phones2phonemes_parser.add_argument(
        "--keep-stress",
        action="store_true",
        help="Keep primary/secondary stress markers",
    )

    # --------
    # coverage
    # --------
    coverage_parser = sub_parsers.add_parser(
        "coverage", help="Calculate coverage of phoneme singletons and pairs"
    )
    coverage_parser.set_defaults(func=do_coverage)

    # -----------------
    # phonemize-lexicon
    # -----------------
    phonemize_lexicon_parser = sub_parsers.add_parser(
        "phonemize-lexicon",
        help="Read a CMU dict-like lexicon and phonemize pronunciations",
    )
    phonemize_lexicon_parser.set_defaults(func=do_phonemize_lexicon)
    phonemize_lexicon_parser.add_argument(
        "--keep-stress",
        action="store_true",
        help="Keep primary/secondary stress markers",
    )

    # ----------------
    # Shared arguments
    # ----------------
    for sub_parser in [
        tokenize_parser,
        phonemize_parser,
        phones2phonemes_parser,
        coverage_parser,
        phonemize_lexicon_parser,
    ]:
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
