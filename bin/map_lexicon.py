#!/usr/bin/env python3
"""Map lexicon from one phoneme set to another"""
import argparse
import re
import sys

_WHITESPACE = re.compile(r"[\t ]+")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="map_lexicon.py")
    parser.add_argument("map", help="Path to mapping file with <from> <to> lines")
    parser.add_argument("--drop", default="_", help="Drop <to> (default: _)")
    args = parser.parse_args()

    phone_map = {}
    with open(args.map, "r", encoding="utf-8") as map_file:
        for line in map_file:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue

            from_phone, to_phone = _WHITESPACE.split(line, maxsplit=1)
            phone_map[from_phone] = to_phone

    # Read lexicon from stdin, output to stdout
    for line_index, line in enumerate(sys.stdin):
        line = line.strip()
        if not line:
            continue

        word, *phones = _WHITESPACE.split(line)
        mapped_phones = []
        for phone in phones:
            mapped_phone = phone_map.get(phone)
            if not mapped_phone:
                for c in phone:
                    if c not in phone_map:
                        print(
                            "Multiple words found on line",
                            line_index + 1,
                            line,
                            file=sys.stderr,
                        )
                        break

            assert (
                mapped_phone
            ), f"Invalid phone on line {line_index + 1}: {phone} {line}"

            if mapped_phone != args.drop:
                mapped_phones.append(mapped_phone)

        if mapped_phones:
            print(word, *mapped_phones)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
