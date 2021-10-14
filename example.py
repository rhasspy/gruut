"""Examples from README"""
from gruut import sentences

text = 'He wound it around the wound, saying "I read it was $10 to read."'

for sent in sentences(text, lang="en-us"):
    for word in sent:
        if word.phonemes:
            print(word.text, *word.phonemes)

# He h ˈi
# wound w ˈaʊ n d
# it ˈɪ t
# around ɚ ˈaʊ n d
# the ð ə
# wound w ˈu n d
# , |
# saying s ˈeɪ ɪ ŋ
# I ˈaɪ
# read ɹ ˈɛ d
# it ˈɪ t
# was w ə z
# ten t ˈɛ n
# dollars d ˈɑ l ɚ z
# to t ə
# read ɹ ˈi d
# . ‖

# -----------------------------------------------------------------------------

print("\neSpeak:")

for sent in sentences(text, lang="en-us", espeak=True):
    for word in sent:
        if word.phonemes:
            print(word.text, *word.phonemes)

# eSpeak:
# He h iː
# wound w ˈa ʊ n d
# it ɪ ɾ
# around ɚ ɹ ˈa ʊ n d
# the ð ə
# wound w ˈuː n d
# , |
# saying s ˈe ɪ ɪ ŋ
# I ˈa ɪ
# read ɹ ˈɛ d
# it ɪ ɾ
# was w ʌ z
# ten t ˈɛ n
# dollars d ˈɑː l ɚ z
# to t ə
# read ɹ ˈiː d
# . ‖

# -----------------------------------------------------------------------------

print("\nSSML:")
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
