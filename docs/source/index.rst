.. gruut documentation master file

gruut
=====

A tokenizer and `IPA <https://en.wikipedia.org/wiki/International_Phonetic_Alphabet>`_ phonemizer for multiple human languages.

.. code-block:: python

    from gruut import text_to_phonemes

    text = 'He wound it around the wound, saying "I read it was $10 to read."'

    for sent_idx, word, word_phonemes in text_to_phonemes(text, lang="en-us"):
        print(word, *word_phonemes)


Output::

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


Note that "wound" and "read" have different pronunciations in different contexts (see :ref:`features` for more details).

Installation
------------

To install gruut with U.S. English support only::

    pip install gruut


Additional languages can be added during installation. For example, with French and Italian support::

    pip install gruut[fr,it]


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
The :py:meth:`gruut.text_to_phonemes` method does everything for you. See the :py:class:`~gruut.TextToPhonemesReturn` enum for ways to adjust the ``return_format``.

If you need more control, see the language-specific classes in :py:mod:`gruut.lang` as well as :py:class:`~gruut.toksen.RegexTokenizer` and :py:class:`~gruut.lang.SqlitePhonemizer`.

Tokenziation operates on text and does the following:

* Splits text into words by whitespace
* Expands user-defined abbreviations
* Breaks apart words and sentences further by punctuation (periods, commas, etc.)
* Drops empty/non-word tokens
* Expands numbers into words (100 -> one hundred)
* Applies upper/lower case filter
* Predicts part of speech tags (see :py:mod:`gruut.pos`)

Once tokenized, phonemization predicts the phonetic pronunciation for each word by:

* Looking up each word in an SQLite database
* Guessing the pronunciation with a pre-trained model (see :py:mod:`gruut.g2p`)

In cases where more than one pronunciation is possible for a word, the "best" pronunciation is:

* Specified by the user with word indexes enabled and a word of the form "word_N" where N is the 1-based pronunciation index
* Whichever pronunciation has the most compatible :ref:`features`.
* The first pronunciation

Command-Line
^^^^^^^^^^^^

gruut tokenization and phonemization can be done externally with a command-line interface.

.. code-block:: bash

   gruut en-us tokenize 'This is a test.' | gruut en-us phonemize | jq -r .pronunciation_text
   ð ˈɪ s ˈɪ z ə t ˈɛ s t ‖

See ``gruut <LANG> <COMMAND> --help`` for more options.

.. _features:

Features
^^^^^^^^

gruut tokens can contain arbitrary features. For now, only part of speech tags are implemented for English and French.

When determining the "best" pronunciation for a word, a phonemizer may consult these features. In English, for example, some word pronunciations in the lexicon contain "preferred" parts of speech. Words like "wind" may be pronounced differently depending on their use as a verb or noun. If a token "wind" is predicted to be a noun during tokenization, then the pronunciation "w ˈɪ n d" is selected instead of "w ˈaɪ n d".

French uses part of speech tags differently. During the post-processing phase of phonemization, these features are help to add liasons between words. For example, in the sentence "J’ai des petites oreilles.", "petites" will be pronounced "p ə t i t z" instead of "p ə t i t".

Inline Pronunciations
^^^^^^^^^^^^^^^^^^^^^

If you want more control over a word's pronunciation, you can include inline pronunciations in your sentences. There are two different syntaxes, with different purposes:

* Brackets - ``[[ p h o n e m e s ]]``
* Curly Braces - ``{{ words with s{eg}m{ent}s }}``

The "brackets" syntax allows you to directly insert phonemes for a word. See `gruut-ipa <https://github.com/rhasspy/gruut-ipa>`_ for the list of phonemes in your desired language. Some substitutions are automatically made for you:

#. Primary and secondary stress can be given with the apostrophe (``'``) and comma (``,``)
#. Elongation can be given with a colon (``:``)
#. Ties will be added, if necessary (e.g., ``tʃ`` becomes ``t͡ʃ``)

The "curly brackets" syntax lets you sound out a word using other words (or segments of other words). For example, "Beyoncé" could be written as ``{{ bee yawn say }}``. From the curly brackets, gruut will look up each word's pronunciation in the lexicon (or guess it), and combine all of the resulting phonemes. You may include phonemes inside the curly brackets as well with the syntax ``/p h o n e m e s/`` alongside other words.

An even more useful aspect of the "curly brackets" syntax is using **word segments**. For most words in the lexicon, gruut has an alignment between its graphemes and phonemes. This enables you do insert *partial* pronunciations of words, such as the "zure" in "azure", with ``a{zure}``. You can even have multiple segments from a single word! For example, ``{{ {mic}roph{one} }}`` will produce phonemes sounding like "mike own".

.. _database:

Database
--------------------------

Word pronunciations and other metadata are stored in SQLite databases with the following tables:

* word_phonemes - word pronunciations
    * id INTEGER - primary key
    * word TEXT - word text
    * pron_order INTEGER - priority of pronunciation (lowest is first)
    * phonemes TEXT - whitespace-separated phonemes
* feature_names - names of extra pronunciation features (like part of speech)
    * id INTEGER - primary key
    * feature_id INTEGER - id of feature (linked to pron_features)
    * feature TEXT - name of feature (e.g., "pos")
* pron_features - extra pronunciation features
    * id INTEGER - primary key
    * pron_id INTEGER - id from word_phonemes
    * feature_id INTEGER - feature_id from feature_names
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

If your lexicon has part of speech (POS) tags, you can add the ``--pos`` flag. In this case, your lexicon must have the following format::

    word POS phoneme ...
    word POS phoneme phoneme ...

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

These additional lexicons can be accessed via the ``model_prefix`` argument to :py:meth:`gruut.get_phonemizer` or the ``--model-prefix`` command-line argument to ``gruut <LANG> phonemize``.

.. code-block:: python

    from gruut import text_to_phonemes

    text = uHe wound it around the wound, saying "I read it was $10 to read."u

    for sent_idx, word, word_phonemes in text_to_phonemes(
        text, lang="en-us", phonemizer_args={"model_prefix": "espeak"}
    ):
        print(word, *word_phonemes)


Output::

    he h ˈiː
    wound w ˈaʊ n d
    it ˈɪ t
    around ɚ ɹ ˈaʊ n d
    the ð ˈə
    wound w ˈuː n d
    , |
    saying s ˈeɪ ɪ ŋ
    i ˈaɪ
    read ɹ ˈɛ d
    it ˈɪ t
    was w ʌ z
    ten t ˈɛ n
    dollars d ˈɑː l ɚ z
    to t ˈuː
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
