# Gruut

A tokenizer, text cleaner, and [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) phonemizer for several human languages that supports [SSML](#ssml).

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

Note that "wound" and "read" have different pronunciations when used in different (grammatical) contexts.

A [subset of SSML](#ssml) is also supported:

```python
from gruut import sentences

ssml_text = """<?xml version="1.0" encoding="ISO-8859-1"?>
<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.w3.org/2001/10/synthesis
                http://www.w3.org/TR/speech-synthesis11/synthesis.xsd"
    xml:lang="en-US">
<s>Today at 4pm, 2/1/2000.</s>
<s xml:lang="it">Un mese fà, 2/1/2000.</s>
</speak>"""

for sent in sentences(ssml_text, ssml=True):
    for word in sent:
        if word.phonemes:
            print(sent.idx, word.lang, word.text, *word.phonemes)
```

with the output:

```
0 en-US Today t ə d ˈeɪ
0 en-US at ˈæ t
0 en-US four f ˈɔ ɹ
0 en-US P p ˈi
0 en-US M ˈɛ m
0 en-US , |
0 en-US February f ˈɛ b j u ˌɛ ɹ i
0 en-US first f ˈɚ s t
0 en-US , |
0 en-US two t ˈu
0 en-US thousand θ ˈaʊ z ə n d
0 en-US . ‖
1 it Un u n
1 it mese ˈm e s e
1 it fà f a
1 it , |
1 it due d j u
1 it gennaio d͡ʒ e n n ˈa j o
1 it duemila d u e ˈm i l a
1 it . ‖
```

See [the documentation](https://rhasspy.github.io/gruut/) for more details.

## Installation

```sh
pip install gruut
```

Languages besides English can be added during installation. For example, with French and Italian support:

```sh
pip install -f 'https://synesthesiam.github.io/prebuilt-apps/' gruut[fr,it]
```

The extra pip repo is needed for an updated [num2words fork](https://github.com/rhasspy/num2words) that includes support for more languages.

You may also [manually download language files](https://github.com/rhasspy/gruut/releases/latest) and use put them in `$XDG_CONFIG_HOME/gruut/` (`$HOME/.config/gruut` by default).

gruut will look for language files in the directory `$XDG_CONFIG_HOME/gruut/<lang>/` if the corresponding Python package is not installed. Note that `<lang>` here is the **full** language name, e.g. `de-de` instead of just `de`. 

## Supported Languages

gruut currently supports:

* Arabic (`ar`)
* Czech (`cs` or `cs-cz`)
* German (`de` or `de-de`)
* English (`en` or `en-us`)
* Spanish (`es` or `es-es`)
* Farsi/Persian (`fa`)
* French (`fr` or `fr-fr`)
* Italian (`it` or `it-it`)
* Dutch (`nl`)
* Russian (`ru` or `ru-ru`)
* Swedish (`sv` or `sv-se`)
* Swahili (`sw`)

The goal is to support all of [voice2json's languages](https://github.com/synesthesiam/voice2json-profiles#supported-languages)

## Dependencies

* Python 3.7 or higher
* Linux
    * Tested on Debian Bullseye
* [num2words fork](https://github.com/rhasspy/num2words) and [Babel](https://pypi.org/project/Babel/)
    * Currency/number handling
    * num2words fork includes additional language support (Arabic, Farsi, Swedish, Swahili)
* gruut-ipa
    * [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) pronunciation manipulation
* [pycrfsuite](https://github.com/scrapinghub/python-crfsuite)
    * Part of speech tagging and grapheme to phoneme models
* [pydateparser](https://github.com/GLibAi/pydateparser)
    * Date parsing for multiple languages

## Numbers, Dates, and More

`gruut` can automatically verbalize numbers, dates, and other expressions. This is done in a locale-aware manner for both parsing and verbalization, so "1/1/2020" may be interpreted as "M/D/Y" or "D/M/Y" depending on the word or sentence's language (e.g., `<s lang="...">`).

The following types of expressions can be automatically expanded into words by `gruut`:

* Numbers - "123" to "one hundred and twenty three" (disable with `verbalize_numbers=False` or `--no-numbers`)
    * Relies on `Babel` for parsing and `num2words` for verbalization
* Dates - "1/1/2020" to "January first, twenty twenty" (disable with `verbalize_dates=False` or `--no-dates`)
    * Relies on `pydateparser` for parsing and both `Babel` and `num2words` for verbalization
* Currency - "$10" to "ten dollars" (disable with `verbalize_currency=False` or `--no-currency`)
    * Relies on `Babel` for parsing and both `Babel` and `num2words` for verbalization
* Times - "12:01am" to "twelve oh one A M" (disable with `verbalize_times=False` or `--no-times`)
    * English only
    * Relies on `num2words` for verbalization

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
  "text_with_ws": "More text.",
  "text_spoken": "More text",
  "par_idx": 0,
  "lang": "en-us",
  "voice": "",
  "words": [
    {
      "idx": 0,
      "text": "More",
      "text_with_ws": "More ",
      "leading_ws": "",
      "training_ws": " ",
      "sent_idx": 0,
      "par_idx": 0,
      "lang": "en-us",
      "voice": "",
      "pos": "JJR",
      "phonemes": [
        "m",
        "ˈɔ",
        "ɹ"
      ],
      "is_major_break": false,
      "is_minor_break": false,
      "is_punctuation": false,
      "is_break": false,
      "is_spoken": true,
      "pause_before_ms": 0,
      "pause_after_ms": 0
    },
    {
      "idx": 1,
      "text": "text",
      "text_with_ws": "text",
      "leading_ws": "",
      "training_ws": "",
      "sent_idx": 0,
      "par_idx": 0,
      "lang": "en-us",
      "voice": "",
      "pos": "NN",
      "phonemes": [
        "t",
        "ˈɛ",
        "k",
        "s",
        "t"
      ],
      "is_major_break": false,
      "is_minor_break": false,
      "is_punctuation": false,
      "is_break": false,
      "is_spoken": true,
      "pause_before_ms": 0,
      "pause_after_ms": 0
    },
    {
      "idx": 2,
      "text": ".",
      "text_with_ws": ".",
      "leading_ws": "",
      "training_ws": "",
      "sent_idx": 0,
      "par_idx": 0,
      "lang": "en-us",
      "voice": "",
      "pos": null,
      "phonemes": [
        "‖"
      ],
      "is_major_break": true,
      "is_minor_break": false,
      "is_punctuation": false,
      "is_break": true,
      "is_spoken": false,
      "pause_before_ms": 0,
      "pause_after_ms": 0
    }
  ],
  "pause_before_ms": 0,
  "pause_after_ms": 0
}
```

For the whole input line and each word, the `text` property contains the processed input text with normalized whitespace while `text_with_ws` retains the original whitespace. The `text_spoken` property only contains words that are spoken, so punctuation and breaks are excluded.

Within each word, there is:

* `idx` - zero-based index of the word in the sentence
* `sent_idx` - zero-based index of the sentence in the input text
* `pos` - part of speech tag (if available)
* `phonemes` - list of [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet) phonemes for the word (if available)
* `is_minor_break` - `true` if "word" separates phrases (comma, semicolon, etc.)
* `is_major_break` - `true` if "word" separates sentences (period, question mark, etc.)
* `is_break` - `true` if "word" is a major or minor break
* `is_punctuation` - `true` if "word" is a surrounding punctuation mark (quote, bracket, etc.)
* `is_spoken` - `true` if not a break or punctuation

See `python3 -m gruut <LANGUAGE> --help` for more options.

### SSML

A subset of [SSML](https://www.w3.org/TR/speech-synthesis11/) is supported:

* `<speak>` - wrap around SSML text
    * `lang` - set language for document
* `<p>` - paragraph
    * `lang` - set language for paragraph
* `<s>` - sentence (disables automatic sentence breaking)
    * `lang` - set language for sentence
* `<w>` / `<token>` - word (disables automatic tokenization)
    * `lang` - set language for word
    * `role` - set word role (see [word roles](#word-roles))
* `<lang lang="...">` - set language inner text
* `<voice name="...">` - set voice of inner text
* `<say-as interpret-as="">` - force interpretation of inner text
    * `interpret-as` one of "spell-out", "date", "number", "time", or "currency"
    * `format` - way to format text depending on `interpret-as`
        * number - one of "cardinal", "ordinal", "digits", "year"
        * date - string with "d" (cardinal day), "o" (ordinal day), "m" (month), or "y" (year)
* `<break time="">` - Pause for given amount of time
    * time - seconds ("123s") or milliseconds ("123ms")
* `<sub alias="">` - substitute `alias` for inner text
* `<phoneme ph="...">` - supply phonemes for inner text
    * `ph` - phonemes for each word of inner text, separated by whitespace
    * `alphabet` - if "ipa", phonemes are intelligently split ("aːˈb" -> "aː", "ˈb")

#### Word Roles

During phonemization, word roles are used to disambiguate pronunciations. Unless manually specified, a word's role is derived from its part of speech tag as `gruut:<TAG>`. For initialisms and `spell-out`, the role `gruut:letter` is used to indicate that e.g., "a" should be spoken as `/eɪ/` instead of `/ə/`.

## Intended Audience

gruut is useful for transforming raw text into phonetic pronunciations, similar to [phonemizer](https://github.com/bootphon/phonemizer). Unlike phonemizer, gruut looks up words in a pre-built lexicon (pronunciation dictionary) or guesses word pronunciations with a pre-trained grapheme-to-phoneme model. Phonemes for each language come from a [carefully chosen inventory](https://en.wikipedia.org/wiki/Template:Language_phonologies).

For each supported language, gruut includes a:

* A word pronunciation lexicon built from open source data
    * See [pron_dict](https://github.com/Kyubyong/pron_dictionaries)
* A pre-trained grapheme-to-phoneme model for guessing word pronunciations

Some languages also include:

* A pre-trained part of speech tagger built from open source data:
    * See [universal dependencies](https://universaldependencies.org/)
