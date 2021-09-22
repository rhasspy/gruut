# Gruut

A tokenizer, text cleaner, and [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) phonemizer for several human languages.

```python
from gruut import sentences

text = 'He wound it around the wound, saying "I read it was $10 to read."'

for sent in sentences(text, lang="en-us"):
    if word.phonemes:
        print(word.text, *word.phonemes)
```

which outputs:

```
He h ˈi
wound w ˈaʊ n d
it ˈɪ t
around ɚ ˈaʊ n d
the ð ə
wound w ˈu n d
, |
saying s ˈeɪ ɪ ŋ
I ˈaɪ
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

A [subset of SSML](#ssml) is also supported:

```python
from gruut import sentences

ssml = '''<speak lang="en_US">
  <s>
    <say-as interpret-as="ordinal">1</say-as> sentence.
  </s>
</speak>
'''

for sent in sentences(text, ssml=True):
    if word.phonemes:
        print(word.lang, word.text, *word.phonemes)
```

See [the documentation](https://rhasspy.github.io/gruut/) for more details.

## Installation

```sh
pip install gruut
```

Additional languages can be added during installation. For example, with French and Italian support:

```sh
pip install gruut[fr,it]
```

You may also [manually download language files](https://github.com/rhasspy/gruut/releases/latest) and use the `--lang-dir` option:

```sh
gruut --language <lang> <text> --lang-dir /path/to/language-files/
```

Extracting the files to `$HOME/.config/gruut/` will allow gruut to automatically make use of them. gruut will look for language files in the directory `$HOME/.config/gruut/<lang>/` if the corresponding Python package is not installed. Note that `<lang>` here is the **full** language name, e.g. `de-de` instead of just `de`. 

## Supported Languages

gruut currently supports:

* Czech (`cs` or `cs-cz`)
* German (`de` or `de-de`)
* English (`en` or `en-us`)
* Spanish (`es` or `es-es`)
* Farsi/Persian (`fa`)
* French (`fr` or `fr-fr`)
* Italian (`it` or `it-it`)
* Dutch (`nl`)
* Russian (`ru` or `ru-ru`)
* Swahili (`sw`)
* Swedish (`sv` or `sv-se`)

The goal is to support all of [voice2json's languages](https://github.com/synesthesiam/voice2json-profiles#supported-languages)

## Dependencies

* Python 3.6 or higher
* Linux
    * Tested on Debian Bullseye
* [num2words fork](https://github.com/rhasspy/num2words) and [Babel](https://pypi.org/project/Babel/)
    * Currency/number handling
    * num2words fork includes additional language support (Arabic, Farsi, Swedish, Swahili)
* gruut-ipa
    * [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) pronunciation manipulation
* [pycrfsuite](https://github.com/scrapinghub/python-crfsuite)
    * Part of speech tagging and grapheme to phoneme models


## Command-Line Usage

The `gruut` module can be executed with `python3 -m gruut --language <LANGUAGE> <TEXT>` or with the `gruut` command (from `setup.py`).

The `gruut` command is line-oriented, consuming text and producing [JSONL](https://jsonlines.org/).
You will probably want to install [jq](https://stedolan.github.io/jq/) to manipulate the [JSONL](https://jsonlines.org/) output from `gruut`.

### Plain Text

Takes raw text and outputs [JSONL](https://jsonlines.org/) with cleaned words/tokens.

```sh
echo 'This, right here, is some "RAW" text!' \
   | gruut --language en-us \
   | jq --raw-output '.words[].text'
This
,
right
here
,
is
some
"
RAW
"
text
!
```

More information is available in the full JSON output:

```sh
gruut --language en-us 'More  text.' | jq .
```

Output:

```json
{
  "idx": 0,
  "text": "More text.",
  "text_with_ws": "More  text.",
  "lang": "en-us",
  "voice": "",
  "words": [
    {
      "idx": 0,
      "text": "More",
      "text_with_ws": "More  ",
      "sent_idx": 0,
      "lang": "en-us",
      "pos": "JJR",
      "phonemes": [
        "m",
        "ˈɔ",
        "ɹ"
      ],
      "is_break": false,
      "is_punctuation": false
    },
    {
      "idx": 1,
      "text": "text",
      "text_with_ws": "text",
      "sent_idx": 0,
      "lang": "en-us",
      "pos": "NN",
      "phonemes": [
        "t",
        "ˈɛ",
        "k",
        "s",
        "t"
      ],
      "is_break": false,
      "is_punctuation": false
    },
    {
      "idx": 2,
      "text": ".",
      "text_with_ws": ".",
      "sent_idx": 0,
      "lang": "en-us",
      "pos": null,
      "phonemes": [
        "‖"
      ],
      "is_break": true,
      "is_punctuation": false
    }
  ]
}
```

For the whole input line and each word, the `text` property contains the processed input text with normalized whitespace while `text_with_ws` retains the original whitespace.

Within each word, there is:

* `idx` - zero-based index of the word in the sentence
* `sent_idx` - zero-based index of the sentence in the input text
* `pos` - part of speech tag (if available)
* `phonemes` - list of [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) phonemes for the word (if available)
* `is_break` - `true` if "word" is a major/minor break (period, comma, etc.)
* `is_punctuation` - `true` if "word" is a punctuation mark (quote, bracket, etc.)

See `python3 -m gruut <LANGUAGE> --help` for more options.

### SSML

A subset of [SSML](https://www.w3.org/TR/speech-synthesis11/) is supported in the input text with:



## Intended Audience

gruut is useful for transforming raw text into phonetic pronunciations, similar to [phonemizer](https://github.com/bootphon/phonemizer). Unlike phonemizer, gruut looks up words in a pre-built lexicon (pronunciation dictionary) or guesses word pronunciations with a pre-trained grapheme-to-phoneme model. Phonemes for each language come from a [carefully chosen inventory](https://en.wikipedia.org/wiki/Template:Language_phonologies).

For each supported language, gruut includes a:

* A word pronunciation lexicon built from open source data
    * See [pron_dict](https://github.com/Kyubyong/pron_dictionaries)
* A pre-trained grapheme-to-phoneme model for guessing word pronunciations

Some languages also include:

* A pre-trained part of speech tagger built from open source data:
    * See [universal dependencies](https://universaldependencies.org/)
