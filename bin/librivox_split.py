#!/usr/bin/env python3
"""Splits Librivox MP3 audio book files into WAV/text fragments according to an Aeneas sync map."""
import argparse
import io
import logging
import json
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

    for mp3_name, mp3_info in book["audio"].items():
        # Load MP3
        mp3_path = input_dir / mp3_name
        audio = AudioSegment.from_mp3(mp3_path)

        sync_path = mp3_path.with_suffix(".json")
        with open(sync_path, "r") as sync_file:
            sync_map = json.load(sync_file)

        mp3_stem = mp3_path.stem
        output_dir = input_dir / mp3_stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write WAV files and transcriptions
        i = 0
        for fragment in sync_map["fragments"]:
            begin_ms = int(float(fragment["begin"]) * 1000)
            end_ms = int(float(fragment["end"]) * 1000)

            if (end_ms - begin_ms) >= args.min_ms:
                with io.StringIO() as text_io:
                    for line in fragment["lines"]:
                        line = line.strip()
                        if line:
                            print(line, file=text_io)

                    text = text_io.getvalue().strip()

                if text:
                    fragment_path = output_dir / f"{mp3_stem}_{i:03d}.wav"

                    # Write text (transcription)
                    text_path = fragment_path.with_suffix(".txt")
                    text_path.write_text(text)

                    # Export audio
                    begin_ms -= args.before_ms
                    end_ms += args.after_ms

                    fragment_audio = audio[begin_ms:end_ms]
                    fragment_audio.export(fragment_path, format="wav")

                    i += 1


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
