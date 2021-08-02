#!/usr/bin/env python3
"""
Uses Aeneas to align Librivox MP3 audio book files with book text.

YAML file format:

---
aeneas:
  language: <AENEAS LANGUAGE>

gruut:
  language: <GRRUT LANGUAGE>

text:
  file: <BOOK TEXT FILE NAME>

audio:
  <MP3 NAME>:
    start_time: <SECONDS TO SKIP FROM START>
    end_time: -<SECONDS TO SKIP FROM END>
    start_line: <LINE TO START IN BOOK TEXT>
    end_line: <LINE TO STOP IN BOOX TEXT>
  <MP3 NAME>:
    ...
"""
import argparse
import logging
from pathlib import Path

import yaml
from aeneas.executetask import ExecuteTask
from aeneas.task import Task

import gruut

_LOGGER = logging.getLogger("librivox_align")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="librivox_align.py")
    parser.add_argument("book_yml", help="YAML file with book details")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.debug(args)

    args.book_yml = Path(args.book_yml)
    input_dir = args.book_yml.parent

    with open(args.book_yml, "r") as book_file:
        book = yaml.safe_load(book_file)

    # Load gruut language
    gruut_tokenizer = gruut.get_tokenizer(book["gruut"]["language"])

    language = book["aeneas"]["language"]

    # Load book text
    text_path = Path(input_dir / book["text"]["file"])
    _LOGGER.debug("Loading book text from %s", text_path)
    with open(text_path, "r") as text_file:
        text = text_file.readlines()

    # Process MP3 files
    for mp3_name, mp3_info in book["audio"].items():
        mp3_path = input_dir / mp3_name
        sync_path = mp3_path.with_suffix(".json")

        config_string = f"task_language={language}|is_text_type=plain|os_task_file_format=json|task_adjust_boundary_no_zero=True"

        start_time = float(mp3_info.get("start_time", 0))
        if start_time > 0:
            # Skip seconds at the beginning
            config_string += f"|is_audio_file_head_length={start_time}"

        end_time = float(mp3_info.get("end_time", 0))
        if end_time < 0:
            # Skip seconds at the end
            end_time = abs(end_time)
            config_string += f"|is_audio_file_tail_length={end_time}"
        elif end_time > 0:
            # Set length of audio
            config_string += f"|is_audio_file_process_length={end_time}"

        task = Task(config_string=config_string)
        task.audio_file_path_absolute = mp3_path.absolute()
        task.sync_map_file_path_absolute = sync_path.absolute()

        mp3_text_path = mp3_path.with_suffix(".txt")
        with open(mp3_text_path, mode="w+") as mp3_text_file:
            start_line = mp3_info.get("start_line", 1)
            end_line = mp3_info.get("end_line", len(text))

            # Clean up newlines in text
            mp3_text = ""
            for line_index in range(start_line - 1, end_line):
                mp3_text += text[line_index].strip() + "\n"

            # Run through gruut tokenizer to expand abbreviations, numbers, etc.
            raw_text_path = mp3_path.with_suffix(".raw.txt")
            with open(raw_text_path, "w") as raw_text_file:
                for sentence in gruut_tokenizer.tokenize(
                    mp3_text, return_format="sentences"
                ):
                    clean_text = " ".join(sentence.clean_words)

                    # Each sentence in on a line now
                    print(clean_text, file=mp3_text_file)
                    print(sentence.raw_text, file=raw_text_file)

            mp3_text_file.seek(0)
            task.text_file_path_absolute = mp3_text_file.name

            # Generate sync map JSON file
            _LOGGER.debug("Generating %s (%s)", sync_path, mp3_path)
            ExecuteTask(task).execute()
            task.output_sync_map_file()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
