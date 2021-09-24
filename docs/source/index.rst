.. gruut documentation master file

gruut
=====

A tokenizer and `IPA <https://en.wikipedia.org/wiki/International_Phonetic_Alphabet>`_ phonemizer for multiple human languages.

.. code-block:: python

    from gruut import sentences

    text = 'He wound it around the wound, saying "I read it was $10 to read."'

    for sent in sentences(text, lang="en-us"):
        if word.phonemes:
            print(word.text, *word.phonemes)

Output::

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


Note that "wound" and "read" have different pronunciations in different contexts.

Installation
------------

To install gruut with U.S. English support only::

    pip install gruut


Additional languages can be added during installation. For example, with French and Italian support::

    pip install -f 'https://synesthesiam.github.io/prebuilt-apps/' gruut[fr,it]

The extra pip repo is needed for an updated `num2words fork <https://github.com/rhasspy/num2words>`_ that includes support for more languages.

Supported Languages
^^^^^^^^^^^^^^^^^^^

* Arabic (``ar``)
* Czech (``cs``)
* German (``de``)
* English (``en``)
* Spanish (``es``)
* French (``fr``)
* Italian (``it``)
* Dutch (``nl``)
* Portuguese (``pt``)
* Russian (``ru``)
* Swedish (``sv``)
* Swahili (``sw``)

Credit to `Michelle K. Hansen <https://www.editions-kawa.com/developpement-personnel/244-a-minute-a-day-to-feel-the-sun-even-if-it-doesnt-shine.html>`_ (translator) for help with German and French language support.


Usage
-----

gruut performs two main functions: tokenization and phonemization.
The :py:meth:`gruut.sentences` method does everything for you, including creating a :py:class:`~gruut.text_processor.TextProcessor` instance for you.

If you need more control, see the language-specific settings in :py:mod:`gruut.lang` and create a :py:class:`gruut.text_processor.TextProcessor` with your custom settings.

Tokenziation operates on text and does the following:

* Splits text into words by whitespace
* Expands user-defined abbreviations and initialisms (TTS/T.T.S.)
* Breaks apart words and sentences further by punctuation (periods, commas, etc.)
* Expands numbers, dates, and currency amounts into words (100 -> one hundred)
* Predicts part of speech tags (see :py:mod:`gruut.pos`)

Once tokenized, phonemization predicts the phonetic pronunciation for each word by:

* Looking up each word in an SQLite database or
* Guessing the pronunciation with a pre-trained model (see :py:mod:`gruut.g2p`)

In cases where more than one pronunciation is possible for a word, the "role" of a word is used to disambiguate. This is normally derived from the word's part of speech (e.g., `gruut:NN`), but can be manually set in SSML with `<w role="...">`.

Command-Line
^^^^^^^^^^^^

gruut tokenization and phonemization can be done externally with a command-line interface.

.. code-block:: bash

   gruut --language en-us 'This is a test.'

which outputs:

.. code-block:: json

    {
        "idx": 0,
        "text": "This is a test.",
        "text_with_ws": "This is a test.",
        "lang": "en-us",
        "voice": "",
        "words": [
            {
            "idx": 0,
            "text": "This",
            "text_with_ws": "This ",
            "sent_idx": 0,
            "lang": "en-us",
            "voice": "",
            "pos": "DT",
            "phonemes": [
                "ð",
                "ˈɪ",
                "s"
            ],
            "is_break": false,
            "is_punctuation": false
            },
            {
            "idx": 1,
            "text": "is",
            "text_with_ws": "is ",
            "sent_idx": 0,
            "lang": "en-us",
            "voice": "",
            "pos": "VBZ",
            "phonemes": [
                "ˈɪ",
                "z"
            ],
            "is_break": false,
            "is_punctuation": false
            },
            {
            "idx": 2,
            "text": "a",
            "text_with_ws": "a ",
            "sent_idx": 0,
            "lang": "en-us",
            "voice": "",
            "pos": "DT",
            "phonemes": [
                "ə"
            ],
            "is_break": false,
            "is_punctuation": false
            },
            {
            "idx": 3,
            "text": "test",
            "text_with_ws": "test",
            "sent_idx": 0,
            "lang": "en-us",
            "voice": "",
            "pos": "NN",
            "phonemes": [
                "t",
                "ˈɛ",
                "s",
                "t"
            ],
            "is_break": false,
            "is_punctuation": false
            },
            {
            "idx": 4,
            "text": ".",
            "text_with_ws": ".",
            "sent_idx": 0,
            "lang": "en-us",
            "voice": "",
            "pos": null,
            "phonemes": [
                "‖"
            ],
            "is_break": true,
            "is_punctuation": false
            }
        ]
    }


See ``gruut --help`` for more options.

Database
--------------------------

Word pronunciations and other metadata are stored in SQLite databases with the following tables:

* word_phonemes - word pronunciations
    * id INTEGER - primary key
    * word TEXT - word text
    * pron_order INTEGER - priority of pronunciation (lowest is first)
    * phonemes TEXT - whitespace-separated phonemes
    * role TEXT - role used to disambiguate pronunciations (e.g., "gruut:NN")
* g2p_alignments - grapheme/phoneme alignments from Phonetisaurus
    * id INTEGER - primary key
    * word TEXT - word from lexicon
    * alignment TEXT - grapheme/phoneme alignment string

You can generate your own lexicon databases from a text file with the format::

    word phoneme ...
    word phoneme phoneme ...
    ...

by simply running::

    python3 -m gruut.lexicon2db --casing lower --lexicon lexicon.txt --database lexicon.db

If your lexicon has word roles, you can add the ``--role`` flag. In this case, your lexicon must have the following format::

    word ROLE phoneme ...
    word ROLE phoneme phoneme ...

Word roles that do not contain a ":" will be formatted as "gruut:<ROLE>".

.. _g2p:

G2P Models
--------------------------

Pre-trained g2p (grapheme to phoneme) models are available for all supported languages.
These models guess pronunciations for unknown words, and are trained on the included lexicon databases using `python-crfsuite <https://github.com/scrapinghub/python-crfsuite>`_.

To train your own model, you will need a lexicon with the format::

    word phoneme ...
    word phoneme phoneme ...
    ...

The first step is to use `Phonetisaurus <https://github.com/AdolfVonKleist/Phonetisaurus>`_ to generate an alignment corpus. If you use the `phonetisaurus Python package <https://pypi.org/project/phonetisaurus/>`_, this can be generated with::

    phonetisaurus train --corpus g2p.corpus --model g2p.fst lexicon.txt

The ``g2p.corpus`` file contains aligned graphemes and phonemes, and is used to train a g2p CRF model with gruut::

    python3 -m gruut.g2p train --corpus g2p.corpus --output g2p/model.crf

You can add the grapheme/phoneme alignments from ``g2p.corpus`` to your lexicon database with::

    python3 -m gruut.corpus2db --corpus g2p.corpus --output lexicon.db

See :py:mod:`gruut.g2p` for more details.

POS Taggers
----------------------

Pre-trained pos (part of speech) taggers are available for English and French.
These models predict the part of speech for each word during tokenization, and are trained from the `Universal Dependencies <https://universaldependencies.org/>`_ using `python-crfsuite <https://github.com/scrapinghub/python-crfsuite>`_.

To train your own model, first download files in `CoNLL-U format <https://universaldependencies.org/format.html>`_ from the `Universal Dependencies treebanks <https://universaldependencies.org>`_ and install the `conllu Python package <https://pypi.org/project/conllu/>`_.

Next, run the training script to generate a CRF model::

    python3 -m gruut.pos train --conllu treebank.conllu --output pos/model.crf

You can change the predicted field with ``--label <FIELD>``, which defaults to xpos.

See :py:mod:`gruut.pos` for more details.

eSpeak Phonemes
----------------------

Most languages include an additional lexicon and pre-trained grapheme to phoneme model with IPA generated from `espeak-ng <https://github.com/espeak-ng/espeak-ng>`_.

.. code-block:: python

    from gruut import sentences

    for sent in sentences(text, lang="en-us", espeak=True):
        for word in sent:
            if word.phonemes:
                print(word.text, *word.phonemes)


Output::

    He h iː
    wound w ˈa ʊ n d
    it ɪ ɾ
    around ɚ ɹ ˈa ʊ n d
    the ð ə
    wound w ˈuː n d
    , |
    saying s ˈe ɪ ɪ ŋ
    I ˈa ɪ
    read ɹ ˈɛ d
    it ɪ ɾ
    was w ʌ z
    ten t ˈɛ n
    dollars d ˈɑː l ɚ z
    to t ə
    read ɹ ˈiː d
    . ‖

To generate your own eSpeak lexicon, first gather a list of words in your target language into a text file (one word per line). Next, use the ``bin/espeak_word.sh`` script to generate the lexicon::

    bin/espeak_word.sh <VOICE> < words.txt > lexicon.txt

where ``<VOICE>`` is the voice name you would pass to ``espeak-ng -v <VOICE>``. After generating the lexicon, see the instructions in :ref:`database` and :ref:`g2p` for creating your own lexicon database and g2p models.


Adding a New Language
---------------------

If you'd like to add a new language to gruut, please follow these steps:

#. Add an IPA phoneme set to `gruut-ipa <https://github.com/rhasspy/gruut-ipa>`_
    * I usually follow `Wikipedia's phonology pages <https://en.wikipedia.org/wiki/Template:Language_phonologies>`_
    * Fewer phonemes is better, as long as you can properly represent word pronunciations
    * Try to find a `Kaldi <https://kaldi-asr.org/>`_ speech to text model for your language and match its phonemes to IPA
    * Try running to ``espeak-ng -v <VOICE> -q --ipa <TEXT>`` to see what eSpeak thinks
#. Create or adapt an IPA pronunciation lexicon
    * Check for an existing IPA lexicon from `ipa-dict <https://github.com/open-dict-data/ipa-dict>`_
    * Try to find a `Kaldi <https://kaldi-asr.org/>`_ speech to text model for your language and use the bin/map_lexicon.py script to re-map phonemes to IPA
    * Extend `wiktionary2dict <https://github.com/rhasspy/wiktionary2dict>`_ and create a lexicon from `Wiktionary <https://www.wiktionary.org/>`_
#. Convert your text lexicon to a database by following the instructions in :ref:`database`
    * Put in data/<language>/lexicon.db
#. Create a grapheme to phoneme (g2p) model by following the instructions in :ref:`g2p`
    * Put in data/<language>/g2p/model.crf
#. Edit ``lang.py`` and:
    * Add your language to ``LANG_ALIASES``
    * Add a tokenizer and phonemizer sub-class for your language
    * Return your tokenizer/phonemizer in ``get_tokenizer`` and ``get_phonemizer`` respectively

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
