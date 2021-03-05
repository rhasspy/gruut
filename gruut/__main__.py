#!/usr/bin/env python3
"""Command-line interface to gruut"""
import argparse
import csv
import dataclasses
import itertools
import json
import logging
import os
import shutil
import sys
import tempfile
import typing
from collections import Counter
from pathlib import Path

import jsonlines
import pydash
import yaml

import gruut_ipa

from .toksen import Token
from .utils import (
    WordPronunciation,
    env_constructor,
    load_lexicon,
    maybe_gzip_open,
    pairwise,
)

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut")

_DIR = Path(__file__).parent

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

    if not args.data_dir:
        # Set to ${XDG_CONFIG_HOME}/gruut or ${HOME}/gruut
        maybe_config_home = os.environ.get("XDG_CONFIG_HOME")
        if maybe_config_home:
            args.data_dir = Path(maybe_config_home) / "gruut"
        else:
            args.data_dir = Path.home() / ".config" / "gruut"

    if args.command == "download":
        # Download language-specific files
        do_download(args)
    else:
        # Directories to search for language-specific data files
        data_dirs = [args.data_dir]

        # All other commands
        env_data_dir = os.environ.get("GRUUT_DATA_DIR")
        if env_data_dir:
            data_dirs.append(Path(env_data_dir))

        # Data directory *next to* gruut
        data_dirs.append(_DIR.parent / "data")

        # Data directory *inside* gruut
        data_dirs.append(_DIR / "data")

        # Search for language-specific data directory
        maybe_lang_dir: typing.Optional[Path] = None

        for data_dir in data_dirs:
            maybe_lang_dir = data_dir / args.language
            if maybe_lang_dir.is_dir():
                config_path = maybe_lang_dir / "language.yml"
                if config_path.is_file():
                    break

        assert maybe_lang_dir, "Language '{args.language}' not found in {data_dirs}"
        setattr(args, "lang_dir", maybe_lang_dir)

        # Load configuration
        config_path = args.lang_dir / "language.yml"
        assert config_path.is_file(), f"Missing {config_path}"

        # Set environment variable for config loading
        os.environ["config_dir"] = str(config_path.parent)
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        args.func(config, args)


# -----------------------------------------------------------------------------


def try_load_language(args, language: typing.Optional[str] = None, **kwargs):
    """Attempt to load a language by code (e.g. en-us)"""
    from . import Language

    if not language:
        language = args.language

    assert language

    _LOGGER.debug("Loading %s from %s", language, args.lang_dir)
    gruut_lang = Language.load(language=language, lang_dir=args.lang_dir, **kwargs)
    assert gruut_lang, f"Language not found: {language} in {args.lang_dir}"

    return gruut_lang


# -----------------------------------------------------------------------------


def do_download(args):
    """
    Downloads and extracts pre-trained model to args.data_dir
    """
    from . import __version__
    from .download import download_file

    lang_dir = args.data_dir / args.language

    _LOGGER.debug("Creating %s", lang_dir)
    lang_dir.mkdir(parents=True, exist_ok=True)

    url = args.url_format.format(lang=args.language, version=__version__)
    with tempfile.NamedTemporaryFile(mode="wb+", suffix=".tar.gz") as lang_file:
        _LOGGER.debug("Downloading %s to %s", url, lang_file.name)
        download_file(url, lang_file.name, f"{args.language}.tar.gz")

        _LOGGER.debug("Extracting %s to %s", lang_file.name, lang_dir)
        shutil.unpack_archive(lang_file.name, lang_dir)

    _LOGGER.info("Successfully downloaded %s to %s", args.language, lang_dir)


# -----------------------------------------------------------------------------


def do_tokenize(config, args):
    """
    Split lines from stdin into sentences, tokenize and clean.

    Prints a line of JSON for each sentence.
    """
    gruut_lang = try_load_language(args, custom_tokenizers=(not args.no_pos))
    tokenizer = gruut_lang.tokenizer

    if args.text:
        # Use arguments
        texts = args.text
    else:
        # Use stdin
        texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in texts:
        line = line.strip()
        if not line:
            continue

        sentences = list(
            tokenizer.tokenize(
                line,
                number_converters=args.number_converters,
                replace_currency=(not args.disable_currency),
                guess_pos=(not args.no_pos),
            )
        )

        if args.split_sentences:
            # One output line per sentence
            for sentence in sentences:
                clean_words = sentence.clean_words
                tokens = sentence.tokens

                if args.exclude_non_words:
                    # Exclude punctuations, etc.
                    clean_words = []
                    tokens = []

                    for token in sentence.tokens:
                        if tokenizer.is_word(token.text):
                            clean_words.append(token.text)
                            tokens.append(token)

                writer.write(
                    {
                        "raw_text": sentence.raw_text,
                        "raw_words": sentence.raw_words,
                        "clean_words": clean_words,
                        "clean_text": tokenizer.token_join.join(clean_words),
                        "sentences": [],
                    }
                )
        else:
            # One output line per input line
            raw_words = []
            clean_words = []
            tokens = []

            for sentence in sentences:
                raw_words.extend(sentence.raw_words)
                clean_words.extend(sentence.clean_words)
                tokens.extend(sentence.tokens)

            if args.exclude_non_words:
                # Exclude punctuations, etc.
                all_tokens = tokens
                clean_words = []
                tokens = []

                for token in all_tokens:
                    if tokenizer.is_word(token.text):
                        clean_words.append(token.text)
                        tokens.append(token)

            writer.write(
                {
                    "raw_text": line,
                    "raw_words": raw_words,
                    "clean_words": clean_words,
                    "tokens": [dataclasses.asdict(t) for t in tokens],
                    "clean_text": tokenizer.token_join.join(clean_words),
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
    gruut_lang = try_load_language(args, custom_tokenizers=False)
    tokenizer = gruut_lang.tokenizer
    phonemizer = gruut_lang.phonemizer
    process_pronunciation = None

    # Load phoneme maps
    phoneme_maps: typing.Dict[str, typing.Dict[str, str]] = {}
    if args.map:
        for map_name in args.map:
            map_path = args.lang_dir / "maps" / (map_name + ".txt")
            _LOGGER.debug("Loading phoneme map %s (%s)", map_name, map_path)
            current_map = {}
            with open(map_path, "r") as map_file:
                for line in map_file:
                    line = line.strip()
                    # Skip blank lines and comments
                    if not line or line.startswith("#"):
                        continue

                    gruut_phoneme, _, mapped_phoneme = line.split(maxsplit=2)
                    current_map[gruut_phoneme] = mapped_phoneme

                    # Automatically add mappings for breaks
                    current_map[gruut_ipa.IPA.BREAK_MINOR] = gruut_ipa.IPA.BREAK_MINOR
                    current_map[gruut_ipa.IPA.BREAK_MAJOR] = gruut_ipa.IPA.BREAK_MAJOR

            phoneme_maps[map_name] = current_map

    # Handle language-specific cases
    if args.language == "fa":
        # Genitive case
        def fa_process_pronunciation(word_pron, token):
            if token.pos == "Ne":
                word_pron = list(word_pron)
                word_pron.append("e̞")

            return word_pron

        process_pronunciation = fa_process_pronunciation

    def process_sentence(sentence_obj):
        token_dicts = sentence_obj.get("tokens")
        if token_dicts:
            tokens = [Token(**t) for t in token_dicts]
        else:
            clean_words = sentence_obj["clean_words"]
            tokens = [Token(text=w) for w in clean_words]

        sentence_prons = phonemizer.phonemize(
            tokens,
            word_indexes=args.word_indexes,
            word_breaks=args.word_breaks,
            process_pronunciation=process_pronunciation,
            use_pos=(not args.no_pos),
        )
        sentence_obj["pronunciations"] = [
            wp.phonemes for word_prons in sentence_prons for wp in word_prons
        ]

        # Pick first pronunciation for each word
        first_pron = []
        for word_prons in sentence_prons:
            if word_prons:
                first_pron.append(word_prons[0].phonemes)

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

        # Map phonemes
        sentence_obj["mapped_phonemes"] = {}
        for map_name, phoneme_map in phoneme_maps.items():
            mapped_phonemes = []
            for word_pron in first_pron:
                mapped_word_phonemes = []

                for phoneme in word_pron:
                    # Exclude stress for now
                    phoneme = gruut_ipa.IPA.without_stress(phoneme)
                    mapped_phoneme = phoneme_map.get(phoneme)
                    if mapped_phoneme:
                        mapped_word_phonemes.append(mapped_phoneme)

                mapped_phonemes.append(mapped_word_phonemes)

            sentence_obj["mapped_phonemes"][map_name] = mapped_phonemes

        # Print back out with extra info
        writer.write(sentence_obj)

    if os.isatty(sys.stdin.fileno()):
        print("Reading tokenize JSONL from stdin...", file=sys.stderr)

    sentence_objs = []
    missing_words = set()

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        sentence_obj = json.loads(line)

        if args.read_all:
            # Store and check for missing words
            sentence_objs.append(sentence_obj)

            for word in sentence_obj["clean_words"]:
                if (word not in phonemizer.lexicon) and tokenizer.is_word(word):
                    missing_words.add(word)
        else:
            # Process immediate
            process_sentence(sentence_obj)

    if sentence_objs:
        # Guess missing words together (faster)
        if missing_words:
            _LOGGER.debug("Guessing pronunciations for %s word(s)", len(missing_words))
            for word, word_pron in phonemizer.predict(missing_words, nbest=1):
                phonemizer.lexicon[word] = [WordPronunciation(word_pron)]

        # Process delayed sentences
        for sentence_obj in sentence_objs:
            process_sentence(sentence_obj)


# -----------------------------------------------------------------------------


def do_phones_to_phonemes(config, args):
    """Transform/group phones in a pronuncation into language phonemes"""
    phonemes_path = Path(pydash.get(config, "language.phonemes"))

    with open(phonemes_path, "r") as phonemes_file:
        phonemes = gruut_ipa.Phonemes.from_text(phonemes_file)

    keep_stress = pydash.get(config, "language.keep_stress", False)

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
            line_phonemes = phonemes.split(line, keep_stress=keep_stress)
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
    gruut_lang = try_load_language(args, preload_lexicon=True)

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
            for p1, p2 in pairwise(word_pron.phonemes):
                p1 = gruut_ipa.IPA.without_stress(p1)
                p2 = gruut_ipa.IPA.without_stress(p2)
                all_pairs.update((p1, p2))

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
            p = gruut_ipa.IPA.without_stress(p)
            single_counts[p] += 1

        # Count phoneme pairs
        for p1, p2 in pairwise(sentence_prons):
            p1 = gruut_ipa.IPA.without_stress(p1)
            p2 = gruut_ipa.IPA.without_stress(p2)

            pair_counts[(p1, p2)] += 1
            all_pairs.add((p1, p2))

    # Check phonemes
    for p in single_counts:
        if p not in phonemes:
            _LOGGER.warning("Extra phoneme: %s", p)

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


def do_optimize_sentences(config, args):
    """Find phonetically rich sentences"""
    from .optimize import get_optimal_sentences

    gruut_lang = try_load_language(args, preload_lexicon=True)
    lexicon = gruut_lang.phonemizer.lexicon

    if args.text:
        # Read from args
        texts = args.text
    else:
        # Read from stdin
        texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading sentences from stdin...")

    # Get optimal sentences
    optimal_sentences = get_optimal_sentences(
        texts,
        gruut_lang,
        lexicon,
        word_breaks=args.word_breaks,
        silence_phone=args.silence_phone,
        max_sentences=args.max_sentences,
        max_passes=args.max_passes,
        min_length=args.min_length,
        max_length=args.max_length,
        cache_file_path=args.cache_file,
        guess_pos=(not args.no_pos),
    )

    # Print results
    writer = jsonlines.Writer(sys.stdout, flush=True)
    writer.write(
        {
            "single_coverage": optimal_sentences.single_coverage,
            "pair_coverage": optimal_sentences.pair_coverage,
            "pair_score": optimal_sentences.pair_score,
            "sentences": [
                {
                    "sentence": dataclasses.asdict(pron_sentence.sentence),
                    "pronunciations": [
                        wp.phonemes
                        for word_prons in pron_sentence.pronunciations
                        for wp in word_prons
                    ],
                }
                for pron_sentence in optimal_sentences.sentences
            ],
        }
    )


# -----------------------------------------------------------------------------


def do_phonemize_lexicon(config, args):
    """Convert phonetic lexicon to phonemic lexicon"""
    casing = None
    if args.casing == "upper":
        casing = str.upper
    elif args.casing == "lower":
        casing = str.lower

    phonemes_path = Path(pydash.get(config, "language.phonemes"))

    with open(phonemes_path, "r") as phonemes_file:
        phonemes = gruut_ipa.Phonemes.from_text(phonemes_file)

    keep_stress = pydash.get(config, "language.keep_stress", False)
    keep_accents = pydash.get(config, "language.keep_accents", False)

    if args.lexicon:
        # Read from file
        lexicon_file = maybe_gzip_open(args.lexicon, "r")
    else:
        # Read from stdin
        lexicon_file = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading lexicon from stdin...")

    lexicon = load_lexicon(lexicon_file, multi_word=args.multi_word)
    unknown_counts = Counter()

    for word, word_prons in lexicon.items():
        if casing:
            word = casing(word)

        for word_pron in word_prons:
            word_pron_str = "".join(word_pron)
            pron_phonemes = phonemes.split(
                word_pron_str, keep_stress=keep_stress, keep_accents=keep_accents
            )
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


def do_compare_phonemes(config, args):
    """Print comparison of two languages' phonemes"""
    gruut_lang1 = try_load_language(args, custom_tokenizers=False)
    gruut_lang2 = try_load_language(
        args, language=args.language2, custom_tokenizers=False
    )

    assert gruut_lang1, f"Unsupported language: {args.language}"
    assert gruut_lang2, f"Unsupported language: {args.language2}"

    phonemes1 = {p.text: p for p in gruut_lang1.phonemes}
    phonemes2 = {p.text: p for p in gruut_lang2.phonemes}
    both_phonemes = sorted(set(itertools.chain(phonemes1.keys(), phonemes2.keys())))

    writer = csv.writer(sys.stdout, delimiter=args.delimiter)
    writer.writerow(
        ("example1", gruut_lang1.language, gruut_lang2.language, "example2")
    )

    for p_text in both_phonemes:
        p1_text = ""
        p1_example = ""
        p2_text = ""
        p2_example = ""

        p1 = phonemes1.get(p_text)
        if p1:
            p1_text = p1.text
            p1_example = p1.example

        p2 = phonemes2.get(p_text)
        if p2:
            p2_text = p2.text
            p2_example = p2.example

        writer.writerow((p1_example, p1_text, p2_text, p2_example))


# -----------------------------------------------------------------------------


def do_mark_heteronyms(config, args):
    """
    Mark words in text with multiple pronunciations (heteronyms)

    Prints text with heteronyms marked.
    """
    gruut_lang = try_load_language(args)

    if args.text:
        # Use arguments
        texts = args.text
    else:
        # Use stdin
        texts = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading text from stdin...", file=sys.stderr)

    for line in texts:
        line = line.strip()
        if not line:
            continue

        sentences = list(
            gruut_lang.tokenizer.tokenize(
                line,
                number_converters=args.number_converters,
                replace_currency=(not args.disable_currency),
            )
        )

        # One output line per input line
        clean_words = []

        for sentence in sentences:
            for word in sentence.clean_words:
                if len(gruut_lang.phonemizer.lexicon.get(word, [])) > 1:
                    clean_words.append(f"{args.start_mark}{word}{args.end_mark}")
                else:
                    clean_words.append(word)

        print(gruut_lang.tokenizer.token_join.join(clean_words))


# -----------------------------------------------------------------------------


def do_check_wavs(config, args):
    """
    Reads CSV with | delimiter from stdin with id|text

    Cleans text and compares WAV duration with text length.

    Prints a line of JSON for each input line.
    """
    import wave

    gruut_lang = try_load_language(args, custom_tokenizers=False)
    tokenizer = gruut_lang.tokenizer

    csv_file = sys.stdin
    csv_dir = Path.cwd()

    if args.csv:
        args.csv = Path(args.csv)
        csv_file = open(args.csv, "r")
        csv_dir = args.csv.parent
    else:
        if os.isatty(sys.stdin.fileno()):
            print("Reading CSV (| delimited) from stdin...", file=sys.stderr)

    reader = csv.reader(csv_file, delimiter="|")
    for row in reader:
        wav_id, text = row[0], row[1]
        wav_path = csv_dir / f"{wav_id}.wav"
        if not wav_path.is_file():
            _LOGGER.warning("Missing WAV file: %s", wav_path)
            continue

        # Get sample rate/width for duration calculation
        with open(wav_path, "rb") as wav_file:
            with wave.open(wav_file) as wav_reader:
                sample_rate = wav_reader.getframerate()
                num_frames = wav_reader.getnframes()

        # WAV duration in milliseconds
        wav_duration = (num_frames / sample_rate) * 1000

        # Clean text
        clean_words = []
        for sentence in tokenizer.tokenize(text):
            clean_words.extend(sentence.clean_words)

        clean_text = tokenizer.token_join.join(clean_words)
        text_length = len(clean_text)

        div_duration = wav_duration / args.denominator
        length_ratio = wav_duration / text_length

        _LOGGER.debug(
            "%s: %s ms, %s char(s), %s, %s; %s",
            id,
            wav_duration,
            text_length,
            div_duration,
            length_ratio,
            text,
        )

        if div_duration < text_length:
            print(id, "too short")

        if length_ratio < args.ratio:
            print(id, "bad duration/length ratio")


# -----------------------------------------------------------------------------


def do_mbrolize(config, args):
    """
    Reads JSONL from stdin with "mapped_phonemes" property.

    Convert mapped phonemes to MBROLA format.

    Combines all input lines into a single output file.
    """
    durations: typing.Dict[str, int] = {}
    durations_path = args.lang_dir / "durations.txt"

    if durations_path.is_file():
        _LOGGER.debug("Loading phoneme duration map from %s", durations_path)
        with open(durations_path, "r") as durations_file:
            for line in durations_file:
                line = line.strip()
                if (not line) or line.startswith("#"):
                    continue

                phoneme, duration_str = line.split(maxsplit=1)
                durations[phoneme] = int(duration_str)

    # Read JSONL from phonemize
    for line_index, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue

        sentence_obj = json.loads(line)
        raw_text = sentence_obj.get("raw_text", "")
        if raw_text:
            print(";", raw_text)

        clean_text = sentence_obj.get("clean_text", "")
        if clean_text:
            print(";", clean_text)

        if raw_text or clean_text:
            print("")

        clean_words = sentence_obj.get("clean_words", [])

        all_mapped_phonemes = sentence_obj.get("mapped_phonemes")
        if not all_mapped_phonemes:
            _LOGGER.warning("No mapped phonemes for line %s", line_index + 1)
            continue

        if args.map:
            # Use named map
            mapped_phonemes = all_mapped_phonemes.get(args.map)
            if not mapped_phonemes:
                _LOGGER.warning(
                    "No phonemes for map %s (line %s)", args.map, line_index + 1
                )
                continue
        else:
            # Use first map
            mapped_phonemes = next(iter(all_mapped_phonemes.values()))

        for word_index, word_pron in enumerate(mapped_phonemes):
            word = ""
            if word_index < len(clean_words):
                word = clean_words[word_index]

            if word:
                print(";", word)

            for phoneme in word_pron:
                duration_ms = durations.get(phoneme, args.default_duration)

                # Convert minor/major breaks to silence phoneme (_)
                if phoneme == gruut_ipa.IPA.BREAK_MINOR:
                    phoneme = args.silence_phoneme
                    duration_ms = args.minor_break_duration
                elif phoneme == gruut_ipa.IPA.BREAK_MAJOR:
                    phoneme = args.silence_phoneme
                    duration_ms = args.major_break_duration

                print(phoneme, duration_ms)

            print("")


# -----------------------------------------------------------------------------


def do_print_phoneme_ids(config, args):
    """
    Print phonemes for a language and associated integer ids.
    """
    gruut_lang = try_load_language(args, custom_tokenizers=False)
    phonemes_list = gruut_lang.id_to_phonemes(
        no_pad=args.no_pad, no_word_break=args.no_word_break
    )

    # <id> <phoneme>
    for phoneme_idx, phoneme in enumerate(phonemes_list):
        print(phoneme_idx, phoneme)


# -----------------------------------------------------------------------------


def do_phonemes2ids(config, args):
    """
    Converts phonemes from phonemize to integer ids.

    Prints JSONL list for each line of input.
    """
    gruut_lang = try_load_language(args, custom_tokenizers=False)
    phonemes_list = gruut_lang.id_to_phonemes(
        no_pad=args.no_pad, no_word_break=args.no_word_break
    )

    phoneme_to_id = {p: i for i, p in enumerate(phonemes_list)}
    _LOGGER.debug(phoneme_to_id)

    writer = jsonlines.Writer(sys.stdout, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        sentence_obj = json.loads(line)
        pronunciation = sentence_obj.get("pronunciation", [])

        phoneme_strs = []
        phoneme_ids = []
        for word_pron in pronunciation:
            for phoneme in word_pron:
                if not phoneme:
                    continue

                if gruut_ipa.IPA.is_stress(phoneme[0]):
                    stress, phoneme = phoneme[0], phoneme[1:]
                    stress_id = phoneme_to_id.get(stress)
                    if stress_id is not None:
                        phoneme_strs.append(stress)
                        phoneme_ids.append(stress_id)
                    else:
                        _LOGGER.warning("No id for stress %s (%s)", stress, line)

                phoneme_id = phoneme_to_id.get(phoneme)
                if phoneme_id is None:
                    _LOGGER.warning("No id for phoneme %s (%s)", phoneme, line)
                    continue

                phoneme_strs.append(phoneme)
                phoneme_ids.append(phoneme_id)

        # Output to console
        _LOGGER.debug(phoneme_strs)
        writer.write(phoneme_ids)


# -----------------------------------------------------------------------------


def do_print_phoneme_counts(config, args):
    """
    Print counts of all phonemes from the lexicon.
    """
    gruut_lang = try_load_language(args)
    writer = jsonlines.Writer(sys.stdout, flush=True)
    phoneme_counts = Counter()

    for phoneme in gruut_lang.phonemes:
        phoneme_counts[phoneme.text] = 0

    for word_prons in gruut_lang.phonemizer.lexicon.values():
        for word_pron in word_prons:
            for phoneme in word_pron:
                if not phoneme:
                    continue

                if gruut_ipa.IPA.is_stress(phoneme[0]):
                    phoneme = phoneme[1:]

                phoneme_counts[phoneme] += 1

    writer.write(phoneme_counts.most_common())


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
    # download
    # --------
    download_parser = sub_parsers.add_parser(
        "download", help="Download and extract pre-trained model"
    )
    download_parser.add_argument(
        "--url-format",
        default="https://github.com/rhasspy/gruut/releases/download/v{version}/{lang}.tar.gz",
        help="Format string for download URL (gets {version} and {lang}, default: Github)",
    )

    # --------
    # tokenize
    # --------
    tokenize_parser = sub_parsers.add_parser(
        "tokenize", help="Sentencize/tokenize raw text, clean, and expand numbers"
    )
    tokenize_parser.add_argument(
        "text", nargs="*", help="Text to tokenize (default: stdin)"
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
    tokenize_parser.add_argument(
        "--split-sentences",
        action="store_true",
        help="Output one line for every sentence",
    )
    tokenize_parser.add_argument(
        "--no-pos", action="store_true", help="Don't guess parts of speech for words"
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
    phonemize_parser.add_argument(
        "--read-all",
        action="store_true",
        help="Read all sentences and guess words before output",
    )
    phonemize_parser.add_argument(
        "--map", action="append", help="Map phonemes according to a named map"
    )
    phonemize_parser.add_argument(
        "--no-pos",
        action="store_true",
        help="Don't use part of speech to resolve pronunciations",
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

    # --------
    # coverage
    # --------
    coverage_parser = sub_parsers.add_parser(
        "coverage", help="Calculate coverage of phoneme singletons and pairs"
    )
    coverage_parser.set_defaults(func=do_coverage)

    # ------------------
    # optimize-sentences
    # ------------------
    optimize_sentences_parser = sub_parsers.add_parser(
        "optimize-sentences", help="Find minimal number of phonetically rich sentences"
    )
    optimize_sentences_parser.add_argument(
        "text", nargs="*", help="Candidate sentences (default: stdin)"
    )
    optimize_sentences_parser.add_argument(
        "--word-breaks",
        action="store_true",
        help="Add word break symbol between each word",
    )
    optimize_sentences_parser.add_argument(
        "--silence-phone",
        action="store_true",
        help="Consider beginning/end of sentence",
    )
    optimize_sentences_parser.add_argument(
        "--max-passes",
        type=int,
        default=10,
        help="Maximum number of optimization passes (default: 10)",
    )
    optimize_sentences_parser.add_argument(
        "--max-sentences",
        type=int,
        help="Maximum number of sentences to keep (default: None)",
    )
    optimize_sentences_parser.add_argument(
        "--min-length",
        type=int,
        help="Minimum number of words in a sentence (default: None)",
    )
    optimize_sentences_parser.add_argument(
        "--max-length",
        type=int,
        help="Maximum number of words in a sentence (default: None)",
    )
    optimize_sentences_parser.add_argument(
        "--cache-file", help="File used to cache sentences and pronunciations"
    )
    optimize_sentences_parser.add_argument(
        "--no-pos", action="store_true", help="Don't guess parts of speech for words"
    )
    optimize_sentences_parser.set_defaults(func=do_optimize_sentences)

    # -----------------
    # phonemize-lexicon
    # -----------------
    phonemize_lexicon_parser = sub_parsers.add_parser(
        "phonemize-lexicon",
        help="Read a CMU dict-like lexicon and phonemize pronunciations",
    )
    phonemize_lexicon_parser.add_argument(
        "lexicon", nargs="?", help="Path to lexicon (default: stdin)"
    )
    phonemize_lexicon_parser.set_defaults(func=do_phonemize_lexicon)
    phonemize_lexicon_parser.add_argument(
        "--casing",
        choices=["lower", "upper", "ignore"],
        default="ignore",
        help="Case transformation to apply to words",
    )
    phonemize_lexicon_parser.add_argument(
        "--multi-word",
        action="store_true",
        help="Split up multiple words before /phonemes/",
    )

    # ----------------
    # compare-phonemes
    # ----------------
    compare_phonemes_parser = sub_parsers.add_parser(
        "compare-phonemes",
        help="Print a text comparison of phonemes from two languages",
    )
    compare_phonemes_parser.set_defaults(func=do_compare_phonemes)
    compare_phonemes_parser.add_argument(
        "language2", help="Second language to compare to first language"
    )
    compare_phonemes_parser.add_argument(
        "--delimiter", default=",", help="Field delimiter"
    )

    # ---------------
    # mark-heteronyms
    # ---------------
    mark_heteronyms_parser = sub_parsers.add_parser(
        "mark-heteronyms",
        help="Mark words in text that have multiple pronunciations (heteronyms)",
    )
    mark_heteronyms_parser.set_defaults(func=do_mark_heteronyms)
    mark_heteronyms_parser.add_argument(
        "text", nargs="*", help="Text to tokenize (default: stdin)"
    )
    mark_heteronyms_parser.add_argument(
        "--start-mark", default="[", help="Mark to add to the start of the word"
    )
    mark_heteronyms_parser.add_argument(
        "--end-mark", default="]", help="Mark to add to the end of the word"
    )
    mark_heteronyms_parser.add_argument(
        "--disable-currency",
        action="store_true",
        help="Disable automatic replacement of currency with words (e.g., $1 -> one dollar)",
    )
    mark_heteronyms_parser.add_argument(
        "--number-converters",
        action="store_true",
        help="Allow number_conv form for specifying num2words converter (cardinal, ordinal, ordinal_num, year, currency)",
    )

    # ----------
    # check-wavs
    # ----------
    check_wavs_parser = sub_parsers.add_parser(
        "check-wavs", help="Read id|text CSV, check WAV duration vs. text length"
    )
    check_wavs_parser.add_argument("--csv", help="Path to CSV file (default: stdin)")
    check_wavs_parser.add_argument(
        "--denominator",
        type=float,
        default=30,
        help="Denominator for length check (default: 30)",
    )
    check_wavs_parser.add_argument(
        "--ratio",
        type=float,
        default=10,
        help="Upper-bound of duration/length ration (default: 10)",
    )
    check_wavs_parser.set_defaults(func=do_check_wavs)

    # --------
    # mbrolize
    # --------
    mbrolize_parser = sub_parsers.add_parser(
        "mbrolize", help="Convert phonemized/mapped output to MBROLA format"
    )
    mbrolize_parser.add_argument(
        "--map", help="Name of phoneme map to use (default: first one)"
    )
    mbrolize_parser.add_argument(
        "--default-duration",
        type=int,
        default=80,
        help="Default duration of phonemes (default: 80ms)",
    )
    mbrolize_parser.add_argument(
        "--minor-break-duration",
        type=int,
        default=150,
        help="Duration of minor breaks (default: 150ms)",
    )
    mbrolize_parser.add_argument(
        "--major-break-duration",
        type=int,
        default=300,
        help="Duration of major breaks (default: 300ms)",
    )
    mbrolize_parser.add_argument(
        "--silence-phoneme", default="_", help="Phoneme used for silence (default: _)"
    )
    mbrolize_parser.set_defaults(func=do_mbrolize)

    # -----------------
    # print-phoneme-ids
    # -----------------
    print_phoneme_ids_parser = sub_parsers.add_parser(
        "print-phoneme-ids", help="Print language phonemes and associated integer ids"
    )
    print_phoneme_ids_parser.add_argument(
        "--no-pad", action="store_true", help="Don't include pad symbol (_)"
    )
    print_phoneme_ids_parser.add_argument(
        "--no-word-break",
        action="store_true",
        help="Don't include word break symbol (#)",
    )
    print_phoneme_ids_parser.set_defaults(func=do_print_phoneme_ids)

    # ------------
    # phonemes2ids
    # ------------
    phonemes2ids_parser = sub_parsers.add_parser(
        "phonemes2ids", help="Convert phonemized output to integer ids"
    )
    phonemes2ids_parser.add_argument(
        "--no-pad", action="store_true", help="Don't include pad symbol (_)"
    )
    phonemes2ids_parser.add_argument(
        "--no-word-break",
        action="store_true",
        help="Don't include word break symbol (#)",
    )
    phonemes2ids_parser.set_defaults(func=do_phonemes2ids)

    # --------------------
    # print-phoneme-counts
    # --------------------
    print_phoneme_counts_parser = sub_parsers.add_parser(
        "print-phoneme-counts", help="Print counts of all phonemes from the lexicon"
    )
    print_phoneme_counts_parser.set_defaults(func=do_print_phoneme_counts)

    # ----------------
    # Shared arguments
    # ----------------
    for sub_parser in [
        download_parser,
        tokenize_parser,
        phonemize_parser,
        phones2phonemes_parser,
        coverage_parser,
        optimize_sentences_parser,
        phonemize_lexicon_parser,
        compare_phonemes_parser,
        mark_heteronyms_parser,
        check_wavs_parser,
        mbrolize_parser,
        print_phoneme_ids_parser,
        phonemes2ids_parser,
        print_phoneme_counts_parser,
    ]:
        sub_parser.add_argument(
            "--data-dir", help="Directory with language-specific data files"
        )
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    return parser.parse_args()


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
