"""Classes for dealing with phones and phonemes"""
import logging
import typing
import unicodedata

from .constants import (  # noqa: F401
    CONSONANTS,
    IPA,
    SCHWAS,
    VOWELS,
    BreakType,
    Consonant,
    Dipthong,
    Schwa,
    Stress,
    Vowel,
    VowelHeight,
    VowelPlacement,
)

_LOGGER = logging.getLogger("gruut")

# -----------------------------------------------------------------------------


class Phone:
    """Single IPA phone with discritics and suprasegmentals"""

    def __init__(
        self,
        letters: str,
        stress: Stress = Stress.NONE,
        is_long: bool = False,
        is_nasal: bool = False,
        diacritics: typing.Optional[typing.Set[str]] = None,
        suprasegmentals: typing.Optional[typing.Set[str]] = None,
    ):
        self.letters: str = letters
        self.stress: Stress = stress
        self.is_long: bool = is_long
        self.is_nasal: bool = is_nasal

        self.diacritics: typing.Set[str] = diacritics or set()
        self.suprasegmentals: typing.Set[str] = suprasegmentals or set()

        # Decompose suprasegmentals and diacritics
        if self.stress == Stress.PRIMARY:
            self.suprasegmentals.add(IPA.STRESS_PRIMARY)
        elif self.stress == Stress.SECONDARY:
            self.suprasegmentals.add(IPA.STRESS_SECONDARY)

        if self.is_long:
            self.suprasegmentals.add(IPA.LONG)

        if self.is_nasal:
            self.diacritics.add(IPA.NASAL)

        self._text: str = ""

        self.vowel: typing.Optional[Vowel] = VOWELS.get(self.letters)
        self.consonant: typing.Optional[Consonant] = CONSONANTS.get(self.letters)
        self.schwa: typing.Optional[Schwa] = SCHWAS.get(self.letters)

    @property
    def text(self) -> str:
        """Get textual representation of phone"""
        if not self._text:
            self._text = ""

            # Pre-letter suprasegmentals
            if self.stress == Stress.PRIMARY:
                self._text += IPA.STRESS_PRIMARY
            elif self.stress == Stress.SECONDARY:
                self._text += IPA.STRESS_SECONDARY

            # Letters
            self._text += self.letters

            # Diacritics
            if self.is_nasal:
                self._text += IPA.NASAL

            # Post-letter suprasegmentals
            if self.is_long:
                self._text += IPA.LONG

        return self._text

    @property
    def is_vowel(self) -> bool:
        """True if phone is a vowel"""
        return self.vowel is not None

    @property
    def is_consonant(self) -> bool:
        """True if phone is a consonant"""
        return self.consonant is not None

    @property
    def is_schwa(self) -> bool:
        """True if phone is a schwa"""
        return self.schwa is not None

    def __repr__(self) -> str:
        return self.text

    @staticmethod
    def from_string(phone_str: str) -> "Phone":
        """Parse phone from string"""
        # Decompose into base and combining characters
        codepoints = unicodedata.normalize("NFD", phone_str)
        kwargs: typing.Dict[str, typing.Any] = {"letters": "", "diacritics": set()}

        for c in codepoints:
            # Check for stress
            if c == IPA.STRESS_PRIMARY:
                kwargs["stress"] = Stress.PRIMARY
            elif c == IPA.STRESS_SECONDARY:
                kwargs["stress"] = Stress.SECONDARY
            elif IPA.is_long(c):
                # Check for elongation
                kwargs["is_long"] = True
            elif IPA.is_nasal(c):
                # Check for nasalation
                kwargs["is_nasal"] = True
            elif IPA.is_bracket(c) or IPA.is_break(c):
                # Skip brackets/syllable breaks
                pass
            elif IPA.is_tie(c):
                # Keep ties in letters
                kwargs["letters"] += c
            elif unicodedata.combining(c) > 0:
                # Stow some diacritics that we don't do anything with
                kwargs["diacritics"].add(c)
            else:
                # Include all other characters in letters
                kwargs["letters"] += c

        return Phone(**kwargs)


class Break:
    """IPA break/boundary"""

    def __init__(self, break_type: BreakType):
        self.type = break_type

        if self.type == BreakType.MINOR:
            self.text = IPA.BREAK_MINOR
        elif self.type == BreakType.MAJOR:
            self.text = IPA.BREAK_MAJOR
        elif self.type == BreakType.WORD:
            self.text = IPA.BREAK_WORD
        else:
            raise ValueError(f"Unrecognized break type: {type}")

    def __repr__(self) -> str:
        return self.text

    @staticmethod
    def from_string(break_str: str) -> "Break":
        """Parse break from string"""
        if break_str == IPA.BREAK_MINOR:
            break_type = BreakType.MINOR
        elif break_str == IPA.BREAK_MAJOR:
            break_type = BreakType.MAJOR
        elif break_str == IPA.BREAK_WORD:
            break_type = BreakType.WORD
        else:
            raise ValueError(f"Unrecognized break type: {break_str}")

        return Break(break_type)


# -----------------------------------------------------------------------------


class Pronunciation:
    """Collection of phones and breaks for some unit of text (word, sentence, etc.)"""

    def __init__(self, phones_and_breaks: typing.List[typing.Union[Phone, Break]]):
        self.phones_and_breaks = phones_and_breaks

        self.phones: typing.List[Phone] = []
        self.breaks: typing.List[Break] = []

        # Decompose into phones and breaks
        for pb in self.phones_and_breaks:
            if isinstance(pb, Phone):
                self.phones.append(pb)
            elif isinstance(pb, Break):
                self.breaks.append(pb)

        self._text = ""

    @property
    def text(self) -> str:
        """Get text representation of pronunciation"""
        if not self._text:
            self._text = "".join(pb.text for pb in self.phones_and_breaks)

        return self._text

    def __repr__(self) -> str:
        return self.text

    def __iter__(self):
        return iter(self.phones_and_breaks)

    @staticmethod
    def from_string(pron_str: str, keep_stress: bool = True) -> "Pronunciation":
        """Split an IPA pronunciation into phones.

        Stress markers bind to the next non-combining codepoint (e.g., ˈa).
        Elongation markers bind to the previous non-combining codepoint (e.g., aː).
        Ties join two non-combining sequences (e.g. t͡ʃ).

        Whitespace and brackets are skipped.

        Returns list of phones.
        """
        clusters = []
        cluster = ""
        skip_next_cluster = False

        codepoints = unicodedata.normalize("NFD", pron_str)

        for codepoint in codepoints:
            new_cluster = False

            if (
                codepoint.isspace()
                or IPA.is_bracket(codepoint)
                or (codepoint in {IPA.BREAK_SYLLABLE})
            ):
                # Skip whitespace, brackets, and syllable breaks
                continue

            if IPA.is_break(codepoint):
                # Keep minor/major/word breaks
                new_cluster = True

            if IPA.is_stress(codepoint):
                if keep_stress:
                    new_cluster = True
                    skip_next_cluster = True
                else:
                    # Drop stress
                    continue
            elif IPA.is_long(codepoint):
                # Add to current cluster
                pass
            elif IPA.is_tie(codepoint):
                # Add to next non-combining to current cluster
                skip_next_cluster = True
            elif unicodedata.combining(codepoint) == 0:
                # Non-combining character
                if skip_next_cluster:
                    # Add to current cluster
                    skip_next_cluster = False
                else:
                    # Start a new cluster
                    new_cluster = True

            if new_cluster and cluster:
                clusters.append(cluster)
                cluster = ""

            cluster += codepoint

        if cluster:
            clusters.append(cluster)

        phones_and_breaks: typing.List[typing.Union[Phone, Break]] = []
        for cluster in clusters:
            if IPA.is_break(cluster):
                phones_and_breaks.append(Break.from_string(cluster))
            else:
                phones_and_breaks.append(Phone.from_string(cluster))

        return Pronunciation(phones_and_breaks)


# -----------------------------------------------------------------------------


class Phoneme:
    """Phoneme composed of international phonetic alphabet symbols"""

    def __init__(self, text: str, example: str = "", unknown: bool = False):
        self._text = ""
        self.example = example
        self.unknown = unknown

        self.stress: Stress = Stress.NONE
        self.elongated: bool = False
        self.nasalated: bool = False
        self._extra_combining: typing.List[str] = []

        # Decompose into base and combining characters
        codepoints = unicodedata.normalize("NFD", text)
        self.letters = ""

        for c in codepoints:
            # Check for stress
            if c == IPA.STRESS_PRIMARY:
                self.stress = Stress.PRIMARY
            elif c == IPA.STRESS_SECONDARY:
                self.stress = Stress.SECONDARY
            elif IPA.is_long(c):
                # Check for elongation
                self.elongated = True
            elif IPA.is_nasal(c):
                # Check for nasalation
                self.nasalated = True
            elif IPA.is_bracket(c) or IPA.is_break(c):
                # Skip brackets/syllable breaks
                pass
            elif c in {IPA.SYLLABIC, IPA.NON_SYLLABIC, IPA.EXTRA_SHORT}:
                # Stow some diacritics that we don't do anything with
                self._extra_combining.append(c)
            else:
                # Include all other characters in base
                self.letters += c

        self.vowel: typing.Optional[Vowel] = VOWELS.get(self.text)
        self.consonant: typing.Optional[Consonant] = CONSONANTS.get(self.text)
        self.schwa: typing.Optional[Schwa] = SCHWAS.get(self.text)
        self.dipthong: typing.Optional[Dipthong] = None

        if (
            (not self.vowel)
            and (not self.consonant)
            and (not self.schwa)
            and (len(self.letters) == 2)
        ):
            # Check if dipthong (two vowels)
            vowel1 = VOWELS.get(self.letters[0])
            vowel2 = VOWELS.get(self.letters[1])
            if vowel1 and vowel2:
                self.dipthong = Dipthong(vowel1, vowel2)

    @property
    def text(self) -> str:
        """Return letters with stress and elongation."""
        if self._text:
            return self._text

        self._text = self.letters
        if self.stress == Stress.PRIMARY:
            self._text = IPA.STRESS_PRIMARY + self._text
        elif self.stress == Stress.SECONDARY:
            self._text = IPA.STRESS_SECONDARY + self._text

        if self.nasalated:
            self._text += IPA.NASAL

        for c in self._extra_combining:
            self._text += c

        if self.elongated:
            self._text += IPA.LONG

        return self._text

    def copy(self) -> "Phoneme":
        """Create a copy of this phonemes"""
        return Phoneme(text=self.text, example=self.example, unknown=self.unknown)

    def __repr__(self) -> str:
        """Return symbol with stress and elongation."""
        return self.text

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        """Return properties of phoneme as a dict"""
        type_name = "Phoneme"
        props: typing.Dict[str, typing.Any] = {
            "text": repr(self),
            "letters": self.letters,
        }

        if self.unknown:
            props["unknown"] = True

        if self.example:
            props["example"] = self.example

        if self.vowel:
            type_name = "Vowel"
            props["height"] = self.vowel.height.value
            props["placement"] = self.vowel.placement.value
            props["rounded"] = self.vowel.rounded
        elif self.consonant:
            type_name = "Consonant"
            props["type"] = self.consonant.type.value
            props["place"] = self.consonant.place.value
            props["voiced"] = self.consonant.voiced
        elif self.dipthong:
            type_name = "Dipthong"
        elif self.schwa:
            type_name = "Schwa"
            props["r_coloured"] = self.schwa.r_coloured

        props["type"] = type_name

        props["nasalated"] = self.nasalated
        props["elongated"] = self.elongated

        return props

    def to_string(self) -> str:
        """Return descriptive string of phoneme"""
        props = self.to_dict()
        type_name = props.get("type", "Phoneme")

        prop_strs = [f"{k}={v}" for k, v in props.items()]

        return f"{type_name}(" + ", ".join(prop_strs) + ")"


# -----------------------------------------------------------------------------


class Phonemes:
    """Set of phonemes and allophones for a language"""

    COMMENT_STR = "#"

    def __init__(self, phonemes=None, ipa_map=None):
        self.phonemes = phonemes or []
        self.ipa_map = ipa_map or {}

        self._ipa_map_sorted = None
        self._phonemes_sorted = None

        self.update()

    @staticmethod
    def from_text(text_file) -> "Phonemes":
        """Load text file with phonemes, examples, and allophones"""
        lang = Phonemes()

        for line in text_file:
            # Remove comments
            line, *_ = line.split(Phonemes.COMMENT_STR, maxsplit=1)
            line = line.strip()
            if line:
                # phoneme [example] [allophone] [allophone]...
                parts = line.split()
                phoneme_ipa = parts[0]
                example = ""

                if len(parts) > 1:
                    example = parts[1]

                if len(parts) > 2:
                    # Map allophone back to phoneme
                    for homophone in parts[2:]:
                        lang.ipa_map[homophone] = phoneme_ipa

                lang.phonemes.append(Phoneme(text=phoneme_ipa, example=example))

        lang.update()

        return lang

    def update(self):
        """Call after modifying phonemes or IPA map to re-sort"""
        # Split map keys and sort by reverse length
        split_ipa_map = [
            ([pb.text for pb in Pronunciation.from_string(k)], v)
            for k, v in self.ipa_map.items()
        ]

        self._ipa_map_sorted = sorted(
            split_ipa_map, key=lambda kv: len(kv[0]), reverse=True
        )

        # Split phonemes and sort by reverse length
        split_phonemes = [
            ([pb.text for pb in Pronunciation.from_string(p.text)], p)
            for p in self.phonemes
        ]

        self._phonemes_sorted = sorted(
            split_phonemes, key=lambda kp: len(kp[0]), reverse=True
        )

    def split(
        self, pron_str: typing.Union[str, Pronunciation], keep_stress: bool = False
    ) -> typing.List[Phoneme]:
        """Split an IPA pronunciation into phonemes"""
        if (not self._ipa_map_sorted) or (not self._phonemes_sorted):
            self.update()

        word_phonemes: typing.List[Phoneme] = []

        if isinstance(pron_str, Pronunciation):
            # Use supplied pronunication
            pron = pron_str
        else:
            # Split string into pronunciation
            pron = Pronunciation.from_string(pron_str, keep_stress=keep_stress)

        # Get text for IPA phones
        ipas = [pb.text for pb in pron]
        num_ipas: int = len(ipas)

        # pylint: disable=C0200
        for ipa_idx in range(len(ipas)):
            ipa = ipas[ipa_idx]
            if ipa is None:
                # Skip replaced piece
                continue

            original_ipa = ipa
            stress = None
            if keep_stress and IPA.is_stress(ipa[0]):
                # Remove stress for comparisons
                stress = ipa[0]
                ipa = ipa[1:]

            # Try to map current IPA
            for src_ipas, dest_ipa in self._ipa_map_sorted:
                if ipa_idx <= (num_ipas - len(src_ipas)):
                    map_match = True
                    # Look forward into sequence
                    for src_idx in range(len(src_ipas)):
                        if src_ipas[src_idx] != ipas[ipa_idx + src_idx]:
                            map_match = False

                    if map_match:
                        ipa = dest_ipa

                        # Replace
                        ipas[ipa_idx] = dest_ipa

                        # Patch ipas to skip replaced pieces beyond first
                        for src_idx in range(1, len(src_ipas)):
                            ipas[ipa_idx + src_idx] = None
                        break

            phoneme_match = False
            for phoneme_ipas, phoneme in self._phonemes_sorted:
                if ipa_idx <= (num_ipas - len(phoneme_ipas)):
                    phoneme_match = True
                    # Look forward into sequence
                    for phoneme_idx in range(len(phoneme_ipas)):
                        if phoneme_ipas[phoneme_idx] != ipas[ipa_idx + phoneme_idx]:
                            phoneme_match = False

                    if phoneme_match:
                        # Successful match
                        if stress:
                            phoneme = phoneme.copy()
                            if stress == IPA.STRESS_PRIMARY:
                                phoneme.stress = Stress.PRIMARY
                            elif stress == IPA.STRESS_SECONDARY:
                                phoneme.stress = Stress.SECONDARY

                        word_phonemes.append(phoneme)

                        # Patch ipas to skip replaced pieces
                        for phoneme_idx in range(1, len(phoneme_ipas)):
                            ipas[ipa_idx + phoneme_idx] = None

                        break

            if not phoneme_match:
                # Add unknown phoneme
                word_phonemes.append(Phoneme(text=original_ipa, unknown=True))

        return word_phonemes
