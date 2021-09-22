from gruut import sentences

text = 'He wound it around the wound, saying "I read it was $10 to read."'

# for sent in sentences(text, lang="en-us"):
#     for word in sent:
#         if word.phonemes:
#             print(word.text, *word.phonemes)

# he h ˈi
# wound w ˈaʊ n d
# it ˈɪ t
# around ɚ ˈaʊ n d
# the ð ə
# wound w ˈu n d
# , |
# saying s ˈeɪ ɪ ŋ
# i ˈaɪ
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

# for sent in sentences(text, lang="en-us", espeak=True):
#     for word in sent:
#         if word.phonemes:
#             print(word.text, *word.phonemes)

# eSpeak:
# he h ˈiː
# wound w ˈaʊ n d
# it ˈɪ t
# around ɚ ɹ ˈaʊ n d
# the ð ˈə
# wound w ˈuː n d
# , |
# saying s ˈeɪ ɪ ŋ
# i ˈaɪ
# read ɹ ˈɛ d
# it ˈɪ t
# was w ʌ z
# ten t ˈɛ n
# dollars d ˈɑː l ɚ z
# to t ˈuː
# read ɹ ˈiː d
# . ‖

# -----------------------------------------------------------------------------

print("\nSSML:")
ssml_text = """<speak lang="en_US">
  <say-as interpret-as="number" format="ordinal">1</say-as> sentence.
  <s lang="de_DE">
    <say-as interpret-as="number" format="ordinal">2</say-as> Satz.
  </s>
  <s lang="fr_FR">
    <say-as interpret-as="number" format="ordinal">3</say-as> phrase.
  </s>
</speak>
"""

for sent in sentences(ssml_text, ssml=True):
    for word in sent:
        if word.phonemes:
            print(sent.idx, word.lang, word.text, *word.phonemes)
