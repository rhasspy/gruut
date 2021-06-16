from gruut import text_to_phonemes

text = 'He wound it around the wound, saying "I read it was $10 to read."'

for sent_idx, word, word_phonemes in text_to_phonemes(text, lang="en-us"):
    print(word, *word_phonemes)

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

for sent_idx, word, word_phonemes in text_to_phonemes(
    text, lang="en-us", phonemizer_args={"model_prefix": "espeak"}
):
    print(word, *word_phonemes)

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
