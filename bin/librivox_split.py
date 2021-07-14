#!/usr/bin/env python3
"""Splits Librivox MP3 audio book files into WAV/text fragments according to an Aeneas sync map."""
import argparse
import io
import json
import logging
from pathlib import Path

import yaml

from pydub import AudioSegment

_LOGGER = logging.getLogger("librivox_split")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="librivox_split.py")
    parser.add_argument("book_yml", help="YAML file with book details")
    parser.add_argument(
        "--before-ms",
        type=int,
        default=0,
        help="Milliseconds before sync to keep (default: 0)",
    )
    parser.add_argument(
        "--after-ms",
        type=int,
        default=0,
        help="Milliseconds after sync to keep (default: 0)",
    )
    parser.add_argument(
        "--min-ms",
        type=int,
        default=400,
        help="Minimum number of milliseconds for a fragment (default: 400)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.debug(args)

    args.book_yml = Path(args.book_yml)
    input_dir = args.book_yml.parent

    with open(args.book_yml, "r") as book_file:
        book = yaml.safe_load(book_file)

    for mp3_name, _mp3_info in book["audio"].items():
        # Load MP3
        mp3_path = input_dir / mp3_name
        audio = AudioSegment.from_mp3(mp3_path)

        sync_path = mp3_path.with_suffix(".json")
        with open(sync_path, "r") as sync_file:
            sync_map = json.load(sync_file)

        # Load text map (clean to raw)
        text_map = {}
        text_path = mp3_path.with_suffix(".txt")
        raw_text_path = mp3_path.with_suffix(".raw.txt")
        if text_path.is_file() and raw_text_path.is_file():
            # Both files had better have the same number of lines
            with open(text_path, "r") as text_file:
                with open(raw_text_path, "r") as raw_text_file:
                    for line, raw_line in zip(text_file, raw_text_file):
                        text_map[line.strip()] = raw_line.strip()

        mp3_stem = mp3_path.stem
        output_dir = input_dir / mp3_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write audio fragments, transcriptions, and sync maps
        i = 0
        for fragment in sync_map["fragments"]:
            begin_sec = float(fragment["begin"])
            begin_ms = int(begin_sec * 1000)

            end_sec = float(fragment["end"])
            end_ms = int(end_sec * 1000)

            if (end_ms - begin_ms) >= args.min_ms:
                with io.StringIO() as text_io:
                    for line in fragment["lines"]:
                        line = line.strip()
                        if line:
                            print(line, file=text_io)

                    clean_text = text_io.getvalue().strip()

                if clean_text:
                    raw_text = text_map.get(clean_text, clean_text)
                    fragment_path = output_dir / f"{mp3_stem}_{i:03d}.mp3"

                    # Write text (transcription)
                    text_path = fragment_path.with_suffix(".txt")
                    text_path.write_text(raw_text)

                    # Write sync map
                    map_path = fragment_path.with_suffix(".json")
                    with open(map_path, "w") as map_file:
                        json.dump(
                            {
                                "begin": args.before_ms / 1000,
                                "end": (args.before_ms / 1000) + (end_sec - begin_sec),
                                "abs_begin": fragment["begin"],
                                "abs_end": fragment["end"],
                                "language": fragment["language"],
                                "clean_text": clean_text,
                                "raw_text": raw_text,
                            },
                            map_file,
                        )

                    # Export audio
                    begin_ms = max(0, begin_ms - args.before_ms)
                    end_ms += args.after_ms

                    fragment_audio = audio[begin_ms:end_ms]
                    fragment_audio.export(fragment_path, format="mp3")

                    i += 1


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
