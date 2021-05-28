# Gruut

A tokenizer, text cleaner, and [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) phonemizer for several human languages.

```python
from gruut.lang import get_tokenizer, get_phonemizer

text = 'He wound it around the wound, saying "I read it was $10 to read."'

tokenizer = get_tokenizer("en-us")
phonemizer = get_phonemizer("en-us")

for sent in tokenizer.tokenize(text):
    print("Raw:", *sent.raw_words)
    print("Clean:", *sent.clean_words)

    print("Phonemes:")
    sent_phonemes = phonemizer.phonemize(sent.tokens)
    for token, phonemes in zip(sent.tokens, sent_phonemes):
        print(token.text, *phonemes)
```

which outputs:

```
Raw: he wound it around the wound , saying i read it was $ 10 to read .
Clean: he wound it around the wound , saying i read it was ten dollars to read .
Phonemes:
he h ˈi
wound w ˈaʊ n d
it ˈɪ t
around ɚ ˈaʊ n d
the ð ə
wound w ˈu n d
, |
saying s ˈeɪ ɪ ŋ
i ˈaɪ
read ɹ ˈɛ d
it ˈɪ t
was w ə z
ten t ˈɛ n
dollars d ˈɑ l ɚ z
to t ə
read ɹ ˈi d
. ‖
```

Note that "wound" and "read" have different pronunciations when used in different contexts.

Includes a pre-trained U.S. English model with part-of-speech/tense aware pronunciations.
[Pre-trained models](https://github.com/rhasspy/gruut/releases/tag/v1.0.0) are also available for the [supported languages](#support-languages).

## Intended Audience

gruut is useful for transforming raw text into phonetic pronunciations, similar to [phonemizer](https://github.com/bootphon/phonemizer). Unlike phonemizer, gruut looks up words in a pre-built lexicon (pronunciation dictionary) or guesses word pronunciations with a pre-trained grapheme-to-phoneme model. Phonemes for each language come from a [carefully chosen inventory](https://en.wikipedia.org/wiki/Template:Language_phonologies).

For each supported language, gruut includes a:

* A word pronunciation lexicon built from open source data
    * See [pron_dict](https://github.com/Kyubyong/pron_dictionaries)
* A pre-trained grapheme-to-phoneme model for guessing word pronunciations

Some languages also include:

* A pre-trained part of speech tagger built from open source data:
    * See [universal dependencies](https://universaldependencies.org/)

## Supported Languages

gruut currently supports:

* U.S. English (`en-us`)
   * Supports part-of-speech aware pronunciations
* U.K. English (`en-gb`)
* Dutch (`nl`)
* Czech (`cs-cz`)
* German (`de-de`)
* French (`fr-fr`)
* Italian (`it-it`)
* Spanish (`es-es`)
* Russian (`ru-ru`)
* Swedish (`sv-se`)
* Greek (`el-gr`)
* Persian (`fa`)
* Vietnamese (`vi-n`)

The goal is to support all of [voice2json's languages](https://github.com/synesthesiam/voice2json-profiles#supported-languages)

## Dependencies

* Python 3.7 or higher
* Linux
    * Tested on Debian Buster
* [Babel](https://pypi.org/project/Babel/) and [num2words](https://pypi.org/project/num2words/)
    * Currency/number handling
* gruut-ipa
    * [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) pronunciation manipulation
* [pycrfsuite](https://github.com/scrapinghub/python-crfsuite)
    * Part of speech tagging and grapheme to phoneme models

## Installation

```sh
$ pip install gruut
```

For Raspberry Pi (ARM), you will first need to [manually install phonetisaurus](https://github.com/rhasspy/phonetisaurus-pypi/releases).

## Language Download

[Pre-trained models](https://github.com/rhasspy/gruut/releases/tag/v0.8.0) for gruut can be downloaded with:

```sh
$ python3 -m gruut <LANGUAGE> download
```

A U.S. English model is included in the distribution.

By default, models are stored in `$HOME/.config/gruut` (technically `$XDG_CONFIG_HOME/.gruut`). This can be overridden by passing a `--lang-dir` argument to all `gruut` commands.

## Command-Line Usage

The `gruut` module can be executed with `python3 -m gruut <LANGUAGE> <COMMAND> <ARGS>`

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
