#!/usr/bin/env python3
"""Generate IPA lexicon for a list of words using espeak-ng"""
import argparse
import ctypes
import logging
import os
import re
import sys
import typing

from gruut_ipa import Pronunciation

_LOGGER = logging.getLogger("espeak_word")

VOICE_ALIASES = {
    "cs-cz": "cs",
    "de-de": "de",
    "es-es": "es",
    "fr-fr": "fr",
    "it-it": "it",
    "ru-ru": "ru",
    "sv-se": "sv",
}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="espeak_word.py")
    parser.add_argument("language", help="eSpeak voice/language")
    parser.add_argument(
        "--pos",
        action="store_true",
        help="Add pronunciations for different parts of speech (English only)",
    )
    parser.add_argument(
        "--empty-pos-tag",
        default="_",
        help="POS tag to use for default pronunciation (default: _)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of words to process at a time per thread",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if os.isatty(sys.stdin.fileno()):
        print("Reading words from stdin...", file=sys.stderr)

    words = filter(None, map(str.strip, sys.stdin))

    # prompt, POS tag
    prompt_pos: typing.List[typing.Tuple[str, typing.Optional[str]]] = [("", None)]

    if args.pos:
        # Use an initial word to prime eSpeak to change pronunciation for part of speech.
        # Obviously, this only works for English.
        prompt_pos.extend([("preferably", "VB"), ("a", "NN"), ("had", "VBD")])

    voice = VOICE_ALIASES.get(args.language, args.language)
    phonemizer = Phonemizer(default_voice=voice)
    for word in words:
        default_pron: typing.Optional[typing.List[str]] = None

        for prompt, pos_tag in prompt_pos:
            if prompt:
                text = f"{prompt}|{word}|{word}"
                ipa_strs = phonemizer.phonemize(text).split()
                if len(ipa_strs) < 3:
                    # eSpeak combined words
                    _LOGGER.warning("Words were combined: %s -> %s", text, ipa_strs)
                    continue

                ipa_str = ipa_strs[1]
            else:
                text = f"{word}|{word}"
                ipa_strs = phonemizer.phonemize(text).split()
                if len(ipa_strs) < 2:
                    # eSpeak combined words
                    _LOGGER.warning("Words were combined: %s -> %s", text, ipa_strs)
                    continue

                ipa_str = ipa_strs[0]

            ipa_pron = Pronunciation.from_string(ipa_str, keep_stress=True)
            word_pron = [p.text for p in ipa_pron]

            if not word_pron:
                continue

            if args.pos:
                if not prompt:
                    default_pron = word_pron
                elif word_pron == default_pron:
                    # Skip duplicate pronunciation
                    continue

            if args.pos:
                # word pos phonemes
                print(word, pos_tag or args.empty_pos_tag, *word_pron)
            else:
                # word phonemes
                print(word, *word_pron)


# -----------------------------------------------------------------------------


class Phonemizer:
    """Use ctypes and libespeak-ng to get IPA phonemes from text"""

    SEEK_SET = 0

    EE_OK = 0

    AUDIO_OUTPUT_SYNCHRONOUS = 0x02
    espeakPHONEMES_IPA = 0x02
    espeakCHARS_AUTO = 0
    espeakPHONEMES = 0x100

    LANG_SWITCH_FLAG = re.compile(r"\([^)]+\)")

    def __init__(
        self,
        default_voice: typing.Optional[str] = None,
        clause_breakers: typing.Optional[typing.Collection] = None,
    ):
        self.default_voice = default_voice
        self.clause_breakers = clause_breakers or {",", ";", ":", ".", "!", "?"}

        self.libc = ctypes.cdll.LoadLibrary("libc.so.6")
        self.libc.open_memstream.restype = ctypes.POINTER(ctypes.c_char)

        self.lib_espeak = ctypes.cdll.LoadLibrary("libespeak-ng.so")
        sample_rate = self.lib_espeak.espeak_Initialize(
            Phonemizer.AUDIO_OUTPUT_SYNCHRONOUS, 0, None, 0
        )
        assert sample_rate > 0, "Failed to initialize libespeak-ng"

    def phonemize(
        self,
        text: str,
        voice: typing.Optional[str] = None,
        keep_clause_breakers: bool = False,
    ) -> str:
        """Return IPA string for text"""
        voice = voice or self.default_voice

        if voice is not None:
            voice_bytes = voice.encode("utf-8")
            result = self.lib_espeak.espeak_SetVoiceByName(voice_bytes)
            assert result == Phonemizer.EE_OK, f"Failed to set voice to {voice}"

        missing_breakers = []
        if keep_clause_breakers and self.clause_breakers:
            missing_breakers = [c for c in text if c in self.clause_breakers]

        # Create in-memory file for phoneme trace.
        # espeak_TextToPhonemes segfaults no matter what I do, so this is the back.
        phonemes_buffer = ctypes.c_char_p()
        phonemes_size = ctypes.c_size_t()
        phonemes_file = self.libc.open_memstream(
            ctypes.byref(phonemes_buffer), ctypes.byref(phonemes_size)
        )

        try:
            self.lib_espeak.espeak_SetPhonemeTrace(
                Phonemizer.espeakPHONEMES_IPA, phonemes_file
            )

            identifier = ctypes.c_uint()
            user_data = ctypes.c_void_p()
            text_bytes = text.encode("utf-8")
            self.lib_espeak.espeak_Synth(
                text_bytes,
                0,  # buflength
                0,  # position
                0,  # position_type
                0,  # end_position
                Phonemizer.espeakCHARS_AUTO | Phonemizer.espeakPHONEMES,
                identifier,
                user_data,
            )
            self.libc.fflush(phonemes_file)

            phoneme_lines = ctypes.string_at(phonemes_buffer).decode().splitlines()

            # Remove language switching flags, e.g. (en)
            phoneme_lines = [
                Phonemizer.LANG_SWITCH_FLAG.sub("", line) for line in phoneme_lines
            ]

            # Re-insert clause breakers
            if missing_breakers:
                # pylint: disable=consider-using-enumerate
                for line_idx in range(len(phoneme_lines)):
                    if line_idx < len(missing_breakers):
                        phoneme_lines[line_idx] += missing_breakers[line_idx]

            return " ".join(line.strip() for line in phoneme_lines)
        finally:
            self.libc.fclose(phonemes_file)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
