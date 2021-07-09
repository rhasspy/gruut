#!/usr/bin/env python3
import argparse
import csv
import sys

from gruut.lang import get_tokenizer

parser = argparse.ArgumentParser(prog="clean_metadata.py")
parser.add_argument("lang", help="Language code")
parser.add_argument(
    "--has-speaker", action="store_true", help="CSV input has format id|speaker|text"
)
args = parser.parse_args()

tokenizer = get_tokenizer(args.lang)

writer = csv.writer(sys.stdout, delimiter="|")
for row in csv.reader(sys.stdin, delimiter="|"):
    speaker = None

    if args.has_speaker:
        utt_id, speaker, text = row[0], row[1], row[2]
    else:
        utt_id, text = row[0], row[1]

    words = []
    for sentence in tokenizer.tokenize(text):
        words.extend(word for word in sentence.clean_words if tokenizer.is_word(word))

    if args.has_speaker:
        writer.writerow((utt_id, speaker, " ".join(words)))
    else:
        writer.writerow((utt_id, " ".join(words)))
