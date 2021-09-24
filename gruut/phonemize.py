"""Class for getting phonetic pronunciations for tokenized text"""
import itertools
import logging
import sqlite3
import typing
from pathlib import Path

from gruut.const import PHONEMES_TYPE

# -----------------------------------------------------------------------------

_LOGGER = logging.getLogger("gruut.phonemize")

ROLE_TO_PHONEMES = typing.Dict[str, PHONEMES_TYPE]

WORD_TRANSFORM_TYPE = typing.Callable[[str], str]


# -----------------------------------------------------------------------------


class SqlitePhonemizer:
    """Phonemizes text using a lexicon from a sqlite database"""

    DEFAULT_ROLE: str = ""

    def __init__(
        self,
        db_conn: sqlite3.Connection,
        lexicon: typing.Optional[typing.Dict[str, ROLE_TO_PHONEMES]] = None,
        g2p_model: typing.Optional[typing.Dict[str, typing.Union[str, Path]]] = None,
        word_transform_funcs: typing.Optional[
            typing.Iterable[WORD_TRANSFORM_TYPE]
        ] = None,
        # guess_transform_func=None,
    ):
        self.db_conn = db_conn

        # word -> role -> [phonemes]
        self.lexicon = lexicon if lexicon is not None else {}

        # [functions]
        self.word_transform_funcs = word_transform_funcs or []

    def __call__(
        self, word: str, role: typing.Optional[str] = None
    ) -> typing.Optional[PHONEMES_TYPE]:
        # Look up in cache first
        role_to_word = self.lexicon.get(word)

        if role_to_word is not None:
            if role is not None:
                # Exact role
                phonemes = role_to_word.get(role)
                if phonemes is not None:
                    return phonemes

            # Default role
            phonemes = role_to_word.get(SqlitePhonemizer.DEFAULT_ROLE)
            if phonemes is not None:
                return phonemes

            # Any role
            if role_to_word:
                return next(iter(role_to_word.values()))

            # Not in lexicon (or database) for sure because role_to_word was present.
            return None

        transforms = itertools.chain([None], self.word_transform_funcs)
        for transform_func in transforms:
            if transform_func is not None:
                lookup_word = transform_func(word)
            else:
                # No transform
                lookup_word = word

            if not lookup_word:
                continue

            # Load pronunciations for word from database.
            #
            # Ordered by pronunciation descending because so duplicate roles
            # will be overwritten by earlier pronunciation.
            cursor = self.db_conn.execute(
                "SELECT role, phonemes FROM word_phonemes WHERE word = ? ORDER BY pron_order DESC",
                (lookup_word,),
            )

            for row in cursor:
                if role_to_word is None:
                    # Create new lexicon entry for original word
                    role_to_word = {}
                    self.lexicon[word] = role_to_word

                db_role, db_phonemes = row[0], row[1].split()
                role_to_word[db_role] = db_phonemes

            if role_to_word is not None:
                # Link to transformed word
                self.lexicon[lookup_word] = self.lexicon[word]

                # Successfully looked up in the database
                return self(word, role=role)

        # Not in lexicon
        return None
