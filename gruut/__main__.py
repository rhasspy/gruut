#!/usr/bin/env python3
"""Command-line interface to gruut"""
import argparse
import dataclasses
import json
import logging
import os
import sys
import typing
from pathlib import Path

import jsonlines
from gruut_ipa import IPA

from gruut import get_text_processor
from gruut.lang import resolve_lang
from gruut.utils import find_lang_dir

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut")

# Path to gruut base directory
_DIR = Path(__file__).parent

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        # Print known languages and exit
        from gruut.lang import KNOWN_LANGS

        print("Languages:", *sorted(list(KNOWN_LANGS)))
        sys.exit(0)
    elif sys.argv[1] == "--version":
        from gruut import __version__

        print(__version__)
        sys.exit(0)

    args = get_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    if args.lang_dir:
        args.lang_dir = Path(args.lang_dir)

    # -------------------------------------------------------------------------

    text_processor = get_text_processor(
        default_lang=args.language,
        languages=[args.language] if not args.ssml else None,
        load_pos_tagger=(not args.no_pos),
        load_phoneme_lexicon=(not args.no_lexicon),
        load_g2p_guesser=(not args.no_g2p),
        model_prefix=args.model_prefix,
        lang_dir=args.lang_dir,
    )

    if args.debug:
        _LOGGER.debug(text_processor.settings)

    if args.text:
        # Use arguments
        lines = args.text
    else:
        # Use stdin
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in lines:
        if not line.strip():
            continue

        # Only strip off line endings (preserve other whitespace)
        line = line.rstrip("\r\n")
        graph, root = text_processor(
            line,
            ssml=args.ssml,
            pos=(not args.no_pos),
            phonemize=(not (args.no_lexicon and args.no_g2p)),
        )

        if args.debug:
            text_processor.print_graph(
                graph,
                root,
                print_func=lambda *print_args: _LOGGER.debug(
                    " ".join(str(a) for a in print_args)
                ),
            )

        for sentence in text_processor.sentences(
            graph,
            root,
            major_breaks=(not args.no_major_breaks),
            minor_breaks=(not args.no_minor_breaks),
            punctuations=(not args.no_punctuation),
        ):
            sentence_dict = dataclasses.asdict(sentence)
            writer.write(sentence_dict)


# -----------------------------------------------------------------------------


def do_tokenize(args):
    """
    Split lines from stdin into sentences, tokenize and clean.

    Prints a line of JSON for each sentence.
    """
    from gruut.commands import tokenize
    from gruut import get_text_processor

    # tokenizer = get_tokenizer(
    #     args.language,
    #     lang_dir=args.lang_dir,
    #     no_pos=args.no_pos,
    #     use_number_converters=args.number_converters,
    #     do_replace_currency=(not args.disable_currency),
    #     exclude_non_words=(not args.no_exclude_non_words),
    # )
    text_processor = get_text_processor(default_lang=args.language)

    if args.text:
        # Use arguments
        lines = args.text
    else:
        # Use stdin
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in lines:
        line = line.strip()
        if not line:
            continue

        graph, root = text_processor(line)
        if args.debug:
            text_processor.print_graph(graph, root)

        for sentence in text_processor.sentences(graph, root):
            sentence_dict = dataclasses.asdict(sentence)
            writer.write(sentence_dict)

    # for sent_json in tokenize(
    #     tokenizer,
    #     lines,
    #     language=args.language,
    #     is_csv=args.csv,
    #     csv_delimiter=args.csv_delimiter,
    #     split_sentences=args.split_sentences,
    #     inline_pronunciations=args.inline,
    # ):
    #     writer.write(sent_json)


# -----------------------------------------------------------------------------


def do_phonemize(args):
    """
    Reads JSONL from stdin with "clean_words" property.

    Looks up or guesses phonetic pronuncation(s) for all clean words.

    Prints a line of JSON for each input line.
    """
    from .lang import get_phonemizer
    from .commands import phonemize

    word_break = IPA.BREAK_WORD if args.word_breaks else None
    phonemizer = get_phonemizer(
        args.language,
        args.lang_dir,
        use_word_indexes=args.word_indexes,
        word_break=word_break,
        no_g2p=args.no_g2p,
        phonetisaurus_g2p=args.phonetisaurus,
        model_prefix=args.model_prefix,
        inline_pronunciations=args.inline,
    )

    if os.isatty(sys.stdin.fileno()):
        print("Reading tokenize JSONL from stdin...", file=sys.stderr)

    def sentence_generator():
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            yield json.loads(line)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for utt_json in phonemize(phonemizer, sentence_generator()):
        writer.write(utt_json)


# -----------------------------------------------------------------------------


def do_text2phonemes(args):
    """
    Reads text from stdin, outputs JSONL with phonemes.

    Prints a line of JSON for each input line.
    """
    from . import text_to_phonemes, Token

    word_break = IPA.BREAK_WORD if args.word_breaks else None

    if args.text:
        lines = args.text
    else:
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in lines:
        utt_id = ""

        if args.csv:
            # Input format is id|text
            utt_id, line = line.split(args.csv_delimiter, maxsplit=1)

        sentences = text_to_phonemes(
            line,
            lang=args.language,
            inline_pronunciations=args.inline,
            return_format="sentences",
            tokenizer_args={
                "do_replace_currency": (not args.disable_currency),
                "use_number_converters": args.number_converters,
            },
            phonemizer_args={
                "word_break": word_break,
                "use_word_indexes": args.word_indexes,
                "model_prefix": args.model_prefix,
            },
        )

        raw_words: typing.List[str] = []
        clean_words: typing.List[str] = []
        tokens: typing.List[Token] = []

        for sentence in sentences:
            raw_words.extend(sentence.raw_words)
            clean_words.extend(sentence.clean_words)
            tokens.extend(sentence.tokens)

        output = {
            "id": utt_id,
            "raw_text": line,
            "raw_words": raw_words,
            "clean_words": clean_words,
            "tokens": [dataclasses.asdict(t) for t in tokens],
            "clean_text": args.word_separator.join(clean_words),
            "sentences": [dataclasses.asdict(s) for s in sentences],
        }

        pronunciation = []
        for sent_obj in output["sentences"]:
            sent_phonemes = sent_obj["phonemes"]
            sent_obj["pronunciation_text"] = args.phoneme_separator.join(
                phoneme for word_phonemes in sent_phonemes for phoneme in word_phonemes
            )

            pronunciation.extend(sent_phonemes)

        output["pronunciation"] = pronunciation
        output["pronunciation_text"] = args.phoneme_separator.join(
            phoneme for word_phonemes in pronunciation for phoneme in word_phonemes
        )

        writer.write(output)
        sys.stdout.flush()


# -----------------------------------------------------------------------------


def do_phonemes(args):
    """
    Prints phonemes for a given language.
    """
    from gruut.lang import get_phonemizer

    phonemizer = get_phonemizer(
        args.language, args.lang_dir, model_prefix=args.model_prefix
    )

    phonemes = set()

    # Remove stress
    for phoneme in phonemizer.phonemes:
        phonemes.add(IPA.without_stress(phoneme))

    # Print in sorted order
    _LOGGER.debug("Printing phonemes for %s", args.language)
    for phoneme in sorted(phonemes):
        print(phoneme)


# -----------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(prog="gruut")
    parser.add_argument(
        "-l",
        "--language",
        default="en-us",
        help="Default language code (default: en-us)",
    )

    parser.add_argument("text", nargs="*", help="Text to tokenize (default: stdin)")
    parser.add_argument(
        "--ssml", action="store_true", help="Input text is SSML",
    )
    parser.add_argument(
        "--no-numbers",
        action="store_true",
        help="Disable number replacement (1 -> one)",
    )
    parser.add_argument(
        "--no-currency",
        action="store_true",
        help="Disable currency replacement ($1 -> one dollar)",
    )
    parser.add_argument(
        "--no-dates",
        action="store_true",
        help="Disable date replacement (4/1/2021 -> April first twenty twenty one)",
    )
    parser.add_argument(
        "--no-pos", action="store_true", help="Disable part of speech tagger",
    )
    parser.add_argument(
        "--no-lexicon", action="store_true", help="Disable phoneme lexicon database",
    )
    parser.add_argument(
        "--no-g2p", action="store_true", help="Disable grapheme to phoneme guesser",
    )
    parser.add_argument(
        "--no-punctuation",
        action="store_true",
        help="Don't output punctuations (quotes, brackets, etc.)",
    )
    parser.add_argument(
        "--no-major-breaks",
        action="store_true",
        help="Don't output major breaks (periods, question marks, etc.)",
    )
    parser.add_argument(
        "--no-minor-breaks",
        action="store_true",
        help="Don't output minor breaks (commas, semicolons, etc.)",
    )
    parser.add_argument(
        "--model-prefix",
        help="Sub-directory of gruut language data files with different lexicon, etc. (e.g., espeak)",
    )
    parser.add_argument(
        "--lang-dir", help="Directory with language-specific data files"
    )

    # tokenize_parser.add_argument(
    #     "--no-exclude-non-words",
    #     action="store_true",
    #     help="Don't remove punctionation, etc. from clean words",
    # )
    # tokenize_parser.add_argument(
    #     "--split-sentences",
    #     action="store_true",
    #     help="Output one line for every sentence",
    # )

    # parser.add_argument(
    #     "--csv", action="store_true", help="Input format is id|text"
    # )
    # parser.add_argument(
    #     "--csv-delimiter",
    #     default="|",
    #     help="Delimiter between id and text (default: |, requires --csv)",
    # )

    # phonemize_parser.add_argument(
    #     "--fail-on-unknown-words",
    #     action="store_true",
    #     help="Raise an error if there are words whose pronunciations can't be guessed",
    # )
    # phonemize_parser.add_argument(
    #     "--phonetisaurus",
    #     help="Path to Phonetisaurus graph.npz grapheme to phoneme guessing (see bin/fst2npz.py)",
    # )
    # phonemize_parser.add_argument(
    #     "--inline", action="store_true", help="Enable inline phonemes/words in text"
    # )
    # phonemize_parser.add_argument(
    #     "--skip-on-unknown-words",
    #     action="store_true",
    #     help="Skip sentences with words whose pronunciations can't be guessed",
    # )

    # -------------
    # text2phonemes
    # -------------
    # text2phonemes_parser = sub_parsers.add_parser(
    #     "text2phonemes", help="Convert text to phonemes"
    # )
    # text2phonemes_parser.set_defaults(func=do_text2phonemes)
    # text2phonemes_parser.add_argument(
    #     "text", nargs="*", help="Text to tokenize/phonemize (default: stdin)"
    # )
    # text2phonemes_parser.add_argument(
    #     "--word-indexes",
    #     action="store_true",
    #     help="Allow word_n form for specifying nth pronunciation of word from lexicon",
    # )
    # text2phonemes_parser.add_argument(
    #     "--word-breaks",
    #     action="store_true",
    #     help="Add the IPA word break symbol (#) between each word",
    # )
    # text2phonemes_parser.add_argument(
    #     "--model-prefix",
    #     help="Directory to use within default language directory with different lexicon/g2p model (e.g., espeak)",
    # )
    # text2phonemes_parser.add_argument(
    #     "--inline", action="store_true", help="Enable inline phonemes/words in text"
    # )
    # text2phonemes_parser.add_argument(
    #     "--disable-currency",
    #     action="store_true",
    #     help="Disable automatic replacement of currency with words (e.g., $1 -> one dollar)",
    # )
    # text2phonemes_parser.add_argument(
    #     "--number-converters",
    #     action="store_true",
    #     help="Allow number_conv form for specifying num2words converter (cardinal, ordinal, ordinal_num, year, currency)",
    # )
    # text2phonemes_parser.add_argument(
    #     "--word-separator",
    #     default=" ",
    #     help="Separator to add between words in output pronunciation (default: space)",
    # )
    # text2phonemes_parser.add_argument(
    #     "--phoneme-separator",
    #     default=" ",
    #     help="Separator to add between words in output pronunciation (default: space)",
    # )
    # text2phonemes_parser.add_argument(
    #     "--csv", action="store_true", help="Input format is id|text"
    # )
    # text2phonemes_parser.add_argument(
    #     "--csv-delimiter",
    #     default="|",
    #     help="Delimiter between id and text (default: |, requires --csv)",
    # )

    # --------
    # phonemes
    # --------
    # phonemes_parser = sub_parsers.add_parser(
    #     "phonemes", help="Print phonemes for a language"
    # )
    # phonemes_parser.set_defaults(func=do_phonemes)
    # phonemes_parser.add_argument(
    #     "--model-prefix",
    #     help="Directory to use within default language directory with different lexicon (e.g., espeak)",
    # )

    # ----------------
    # Shared arguments
    # ----------------
    # for sub_parser in [
    #     tokenize_parser,
    #     phonemize_parser,
    #     text2phonemes_parser,
    #     phonemes_parser,
    # ]:
    #     sub_parser.add_argument(
    #         "--lang-dir", help="Directory with language-specific data files"
    #     )
    #     sub_parser.add_argument(
    #         "--debug", action="store_true", help="Print DEBUG messages to console"
    #     )

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to console"
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
