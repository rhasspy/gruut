#!/usr/bin/env python3
import csv
import sys

from gruut.lang import get_tokenizer

if len(sys.argv) < 2:
    print("Usage: clean-metadata.py <lang> < CSV > CSV")

lang = sys.argv[1]
tokenizer = get_tokenizer(lang)

writer = csv.writer(sys.stdout, delimiter="|")
for row in csv.reader(sys.stdin, delimiter="|"):
    utt_id, text = row[0], row[1]

    words = []
    for sentence in tokenizer.tokenize(text):
        words.extend(word for word in sentence.clean_words if tokenizer.is_word(word))

    writer.writerow((utt_id, " ".join(words)))
