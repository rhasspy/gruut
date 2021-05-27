import dataclasses
import sys
import typing

from .const import Token
from .phonemize import Phonemizer, UnknownWordError
from .toksen import Tokenizer

# -----------------------------------------------------------------------------


def tokenize(
    tokenizer: Tokenizer,
    lines: typing.Iterable[str],
    is_csv: bool = False,
    csv_delimiter: str = "|",
    split_sentences: bool = False,
) -> typing.Iterable[typing.Dict[str, typing.Any]]:
    for line in lines:
        line = line.strip()
        if not line:
            continue

        utt_id = ""

        if is_csv:
            # Input format is id|text
            utt_id, line = line.split(csv_delimiter, maxsplit=1)

        sentences = list(tokenizer.tokenize(line))

        if split_sentences:
            # One output line per sentence
            for sentence_idx, sentence in enumerate(sentences):
                sentence_id = str(sentence_idx)
                if utt_id:
                    sentence_id = f"{utt_id}_{sentence_id}"

                clean_words = sentence.clean_words
                tokens = sentence.tokens

                yield {
                    "id": sentence_id,
                    "raw_text": sentence.raw_text,
                    "raw_words": sentence.raw_words,
                    "clean_words": clean_words,
                    "clean_text": sentence.clean_text,
                    "sentences": [],
                }

                yield sent_json
        else:
            # One output line per input line
            raw_words = []
            clean_words = []
            tokens = []

            for sentence in sentences:
                raw_words.extend(sentence.raw_words)
                clean_words.extend(sentence.clean_words)
                tokens.extend(sentence.tokens)

            yield {
                "id": utt_id,
                "raw_text": line,
                "raw_words": raw_words,
                "clean_words": clean_words,
                "tokens": [dataclasses.asdict(t) for t in tokens],
                "clean_text": tokenizer.join_str.join(clean_words),
                "sentences": [dataclasses.asdict(s) for s in sentences],
            }


# -----------------------------------------------------------------------------


def phonemize(
    phonemizer: Phonemizer,
    sentence_objects: typing.Iterable[typing.Dict[str, typing.Any]],
    word_separator: str = " ",
    phoneme_separator: str = " ",
    fail_on_unknown_words: bool = False,
    skip_on_unknown_words: bool = False,
) -> typing.Iterable[typing.Dict[str, typing.Any]]:
    for sentence_obj in sentence_objects:
        token_dicts = sentence_obj.get("tokens")
        if token_dicts:
            tokens = [Token(**t) for t in token_dicts]
        else:
            clean_words = sentence_obj["clean_words"]
            tokens = [Token(text=w) for w in clean_words]

        try:
            sentence_pron = list(phonemizer.phonemize(tokens))
            sentence_obj["pronunciation"] = sentence_pron

            # Create string of first pronunciation
            sentence_obj["pronunciation_text"] = word_separator.join(
                phoneme_separator.join(word_pron) for word_pron in sentence_pron
            )

            yield sentence_obj
        except UnknownWordError as e:
            if skip_on_unknown_words:
                _LOGGER.warning(
                    "Skipping utterance %s due to unknown words: %s",
                    sentence_obj.get("id", ""),
                    sentence_obj.get("raw_text", ""),
                )
            elif fail_on_unknown_words:
                # Fail instead of skipping
                raise e


# -----------------------------------------------------------------------------
