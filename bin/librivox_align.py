#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

from aeneas.executetask import ExecuteTask
from aeneas.task import Task

_LOGGER = logging.getLogger("librivox_align")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="librivox_align.py")
    parser.add_argument("input_dir", help="Directory with MP3 and text files")
    parser.add_argument("--lang", required=True, help="Aeneas language")
    args = parser.parse_args()

    args.input_dir = Path(args.input_dir)

    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.debug(args)

    config_string = (
        f"task_language={args.lang}|is_text_type=plain|os_task_file_format=json"
    )

    for mp3_path in args.input_dir.glob("*.mp3"):
        text_path = mp3_path.with_suffix(".txt")
        if not text_path.is_file():
            _LOGGER.warning("Missing %s", text_path)
            continue

        sync_path = mp3_path.with_suffix(".json")
        task = Task(config_string=config_string)

        task.audio_file_path_absolute = mp3_path.absolute()
        task.text_file_path_absolute = text_path.absolute()
        task.sync_map_file_path_absolute = sync_path.absolute()

        _LOGGER.debug("Generating %s (%s %s)", sync_path, mp3_path, sync_path)
        ExecuteTask(task).execute()
        task.output_sync_map_file()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
