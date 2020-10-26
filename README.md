# Gruut

A tokenizer, text cleaner, and phonemizer for many human languages.

Useful for transforming raw text into phonetic pronunciations, similar to [phonemizer](https://github.com/bootphon/phonemizer). Unlike phonemizer, gruut looks up words in a pre-built lexicon (pronunciation dictionary) or guesses word pronunciations with a pre-trained grapheme-to-phoneme model. Phonemes for each language come from a [carefully chosen inventory](https://en.wikipedia.org/wiki/Template:Language_phonologies).

For each supported language, gruut includes a:

* List of [phonemes](https://en.wikipedia.org/wiki/Phoneme) in the [International Phonetic Alphabet](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet)
* Word pronunciation lexicon built from [Wiktionary](https://www.wiktionary.org/)
    * See [pron_dict](https://github.com/Kyubyong/pron_dictionaries)
* Pre-trained [grapheme-to-phoneme model](https://github.com/AdolfVonKleist/Phonetisaurus) for guessing word pronunciations

## Supported Languages

gruut currently supports:

* U.S. English (`en-us`)
* Dutch (`nl`)
* Czech (`cs-cz`)
* German (`de-de`)
* French (`fr-fr`)
* Italian (`it-it`)

The goal is to support all of [voice2json's languages](https://github.com/synesthesiam/voice2json-profiles#supported-languages)

## Dependencies

* Python 3.7 or higher
* Linux
    * Test on Debian Buster
* [Babel](https://pypi.org/project/Babel/) and [num2words](https://pypi.org/project/num2words/)
    * Currency/number handling
* gruut-ipa
    * [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) pronunciation manipulation
* [phonetisaurus](https://github.com/rhasspy/phonetisaurus-pypi)
    * Guessing word pronunciations outside lexicon

## Installation

```sh
$ pip install gruut
```

for Raspberry Pi (ARM), you will first need to [manually install phonetisaurus](https://github.com/rhasspy/phonetisaurus-pypi/releases).

## Usage

The `gruut` module can be executed with `python3 -m gruut <COMMAND> <ARGS>`

The commands are line-oriented, consuming/producing either text or [JSONL](https://jsonlines.org/).
They can be composed to produce a pipeline for cleaning text.

You will probably want to install [jq](https://stedolan.github.io/jq/) to manipulate the [JSONL](https://jsonlines.org/) output from `gruut`.

### tokenize

Takes raw text and outputs [JSONL](https://jsonlines.org/) with cleaned words/tokens.

```sh
$ echo 'This, right here, is some RAW text!' \
    | python3 -m gruut en-us tokenize \
    | jq -c .clean_words
["this", ",", "right", "here", ",", "is", "some", "raw", "text", "!"]
```

See `python3 -m gruut <LANGUAGE> tokenize --help` for more options.

### phonemize

Takes [JSONL](https://jsonlines.org/) output from `tokenize` and produces [JSONL](https://jsonlines.org/) with phonemic pronunciations.

```sh
$ echo 'This, right here, is some RAW text!' \
    | python3 -m gruut en-us tokenize \
    | python3 -m gruut en-us phonemize \
    | jq -c .pronunciation_text
ð ɪ s | ɹ aɪ t h iː ɹ | ɪ z s ʌ m ɹ ɑː t ɛ k s t ‖
```

See `python3 -m gruut <LANGUAGE> phonemize --help` for more options.

### phones2phonemes

Takes IPA pronunciations (one per line) and outputs [JSONL](https://jsonlines.org/) with phonemes and their descriptions.

```sh
$ echo '/ˈt͡ʃuːz/' \
    | python3 -m gruut en-us phones2phonemes --keep-stress \
    | jq .phonemes
[
  {
    "text": "t͡ʃ",
    "letters": "t͡ʃ",
    "example": "[ch]in",
    "stress": "primary",
    "type": "Consonant",
    "place": "post-alveolar",
    "voiced": false,
    "nasalated": false,
    "elongated": false
  },
  {
    "text": "uː",
    "letters": "u",
    "example": "s[oo]n",
    "stress": "none",
    "height": "close",
    "placement": "back",
    "rounded": true,
    "type": "Vowel",
    "nasalated": false,
    "elongated": true
  },
  {
    "text": "z",
    "letters": "z",
    "example": "[z]ing",
    "stress": "none",
    "type": "Consonant",
    "place": "alveolar",
    "voiced": true,
    "nasalated": false,
    "elongated": false
  }
]
```

See `python3 -m gruut <LANGUAGE> phones2phonemes --help` for more options.

## coverage

Takes [JSONL](https://jsonlines.org/) from from `phonemize` and outputs a coverage report for all singleton and phoneme pairs.

```sh
$ echo 'The quick brown fox jumps over the lazy dog.' \
    | python3 -m gruut en-us tokenize \
    | python3 -m gruut en-us phonemize \
    | python3 -m gruut en-us coverage \
    | jq -c .coverage
{"single":0.6190476190476191,"pair":0.021386430678466076}
```

With [multiple sentences](https://www.cs.columbia.edu/~hgs/audio/harvard.html):

```sh
$ cat << EOF |
The birch canoe slid on the smooth planks.
Glue the sheet to the dark blue background.
It's easy to tell the depth of a well.
These days a chicken leg is a rare dish.
Rice is often served in round bowls.
The juice of lemons makes fine punch.
The box was thrown beside the parked truck.
The hogs were fed chopped corn and garbage.
Four hours of steady work faced us.
Large size in stockings is hard to sell.
EOF
    python3 -m gruut en-us tokenize \
    | python3 -m gruut en-us phonemize \
    | python3 -m gruut en-us coverage \
    | jq -c .coverage
{"single":0.8809523809523809,"pair":0.1364306784660767}
```
