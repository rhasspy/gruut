#!/usr/bin/env python3
import argparse
import logging
import json
from pathlib import Path

from pydub import AudioSegment

_LOGGER = logging.getLogger("librivox_split")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="librivox_split.py")
    parser.add_argument("audio_file", help="MP3 audio file")
    parser.add_argument("sync_file", help="Sync JSON map")
    parser.add_argument("output_dir", help="Directory to write audio files")
    parser.add_argument("--prefix", help="Prefix for audio files")
    parser.add_argument(
        "--min-ms",
        type=int,
        default=400,
        help="Minimum number of milliseconds for a fragment (default: 400)",
    )
    args = parser.parse_args()

    args.audio_file = Path(args.audio_file)
    args.sync_file = Path(args.sync_file)
    args.output_dir = Path(args.output_dir)

    if not args.prefix:
        args.prefix = args.audio_file.stem

    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.debug(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load MP3
    audio = AudioSegment.from_mp3(args.audio_file)
    with open(args.sync_file, "r") as sync_file:
        sync_map = json.load(sync_file)

    # Write WAV files and transcriptions
    i = 0
    for fragment in sync_map["fragments"]:
        begin_ms = float(fragment["begin"]) * 1000
        end_ms = float(fragment["end"]) * 1000

        if (end_ms - begin_ms) >= args.min_ms:
            fragment_path = args.output_dir / f"{args.prefix}_{i:03d}.wav"
            fragment_audio = audio[begin_ms:end_ms]
            fragment_audio.export(fragment_path, format="wav")

            text_path = fragment_path.with_suffix(".txt")
            with open(text_path, "w") as text_file:
                for line in fragment["lines"]:
                    line = line.strip()
                    if line:
                        print(line, file=text_file)

            i += 1


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
