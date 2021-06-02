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

* Czech (``cs``)
* German (``de``)
* English (``en``)
* Spanish (``es``)
* Farsi/Persian (``fa``)
* French (``fr``)
* Italian (``it``)
* Dutch (``nl``)
* Russian (``ru``)
* Swedish (``sv``)


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


.. _features:

Features
^^^^^^^^

gruut tokens can contain arbitrary features. For now, only part of speech tags are implemented for English and French.

When determining the "best" pronunciation for a word, a phonemizer may consult these features. In English, for example, some word pronunciations in the lexicon contain "preferred" parts of speech. Words like "wind" may be pronounced differently depending on their use as a verb or noun. If a token "wind" is predicted to be a noun during tokenization, then the pronunciation "w ˈɪ n d" is selected instead of "w ˈaɪ n d".

French uses part of speech tags differently. During the post-processing phase of phonemization, these features are help to add liasons between words. For example, in the sentence "J’ai des petites oreilles.", "petites" will be pronounced "p ə t i t z" instead of "p ə t i t".


G2P Models
--------------------------

Pre-trained g2p (grapheme to phoneme) models are available for all supported languages.
These models guess pronunciations for unknown words, and are trained on the included lexicon databases using `python-crfsuite <https://github.com/scrapinghub/python-crfsuite>`_.

See :py:mod:`gruut.g2p` for more details.

POS Taggers
----------------------

Pre-traned pos (part of speech) taggers are available for English and French.
These models predict the part of speech for each word during tokenization, and are trained from the `Universal Dependencies <https://universaldependencies.org/>`_ using `python-crfsuite <https://github.com/scrapinghub/python-crfsuite>`_.

See :py:mod:`gruut.pos` for more details.

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
