#!/usr/bin/env python3
"""Expands/normalizes text in CSV metadata"""
import argparse
import csv
import sys
import typing

from gruut import sentences

parser = argparse.ArgumentParser(prog="clean_metadata.py")
parser.add_argument("lang", help="Language code")
parser.add_argument(
    "--has-speaker", action="store_true", help="CSV input has format id|speaker|text"
)
args = parser.parse_args()

writer = csv.writer(sys.stdout, delimiter="|")
for row in csv.reader(sys.stdin, delimiter="|"):
    speaker = None

    if args.has_speaker:
        utt_id, speaker, text = row[0], row[1], row[2]
    else:
        utt_id, text = row[0], row[1]

    words: typing.List[str] = []
    for sentence in sentences(text):
        words.extend(word.text for word in sentence if word.is_spoken)

    if args.has_speaker:
        writer.writerow((utt_id, speaker, " ".join(words)))
    else:
        writer.writerow((utt_id, " ".join(words)))
