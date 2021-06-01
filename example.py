from gruut import text_to_phonemes

text = 'He wound it around the wound, saying "I read it was $10 to read."'

for sent_idx, word, word_phonemes in text_to_phonemes(text, lang="en-us"):
    print(word, *word_phonemes)
