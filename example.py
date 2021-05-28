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
