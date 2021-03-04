"""Methods to find phonetically rich sentences for a language"""
import logging
import os
import random
import re
import typing
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import jsonlines
from dataclasses_json import DataClassJsonMixin

from gruut_ipa import IPA

from . import Language
from .toksen import Sentence
from .utils import LEXICON_TYPE, WordPronunciation, pairwise

_LOGGER = logging.getLogger("gruut.optimize")

# Beginning/ending of sentence
_SILENCE_PHONE = "_"

DIPHONE_TYPE = typing.Tuple[str, str]

# -----------------------------------------------------------------------------


@dataclass
class PronouncedSentence(DataClassJsonMixin):
    """Tokenized sentence with pronunciation"""

    sentence: Sentence
    pronunciations: typing.List[typing.List[WordPronunciation]]


@dataclass
class OptimalSentences:
    """Result from get_optimal_sentences"""

    sentences: typing.List[PronouncedSentence]
    single_coverage: float
    pair_coverage: float
    pair_score: float
    pair_counts: typing.Dict[DIPHONE_TYPE, int]


# -----------------------------------------------------------------------------


def get_optimal_sentences(
    text_iter: typing.Iterable[str],
    lang: Language,
    lexicon: LEXICON_TYPE,
    max_sentences: typing.Optional[int] = None,
    max_examples: int = 20,
    max_passes: int = 10,
    silence_phone: bool = False,
    word_breaks: bool = False,
    min_length: typing.Optional[int] = None,
    max_length: typing.Optional[int] = None,
    cache_file_path: typing.Optional[typing.Union[str, Path]] = None,
    keep_stress: bool = False,
    guess_pos: bool = True,
) -> OptimalSentences:
    """Find minimal set of sentences with diphone example coverage"""
    phonemes = set(p.text for p in lang.phonemes)
    if word_breaks:
        phonemes.add(IPA.BREAK_WORD)

    if silence_phone:
        phonemes.add(_SILENCE_PHONE)

    # Get set of phoneme pairs from the lexicon.
    # This is done instead of using all possible phoneme pairs, because there
    # are many pairs that either humans cannot produce or are not used.
    # We assume the lexicon will contain an example of each useful pairs.
    all_pairs = set()
    for word_prons in lexicon.values():
        for word_pron in word_prons:
            all_pairs.update(pairwise(remove_stress(word_pron, keep_stress).phonemes))

    sentences: typing.List[PronouncedSentence] = []

    if cache_file_path and os.path.isfile(cache_file_path):
        _LOGGER.debug("Loading sentences/pronunciations from %s", cache_file_path)
        with open(cache_file_path, "r") as cache_file:
            for line in cache_file:
                line = line.strip()
                if line:
                    sentences.append(PronouncedSentence.from_json(line))

        _LOGGER.debug("Loaded %s sentence(s)", len(sentences))
    else:
        # Load sentences
        clean_sentences: typing.List[Sentence] = []
        missing_words: typing.Set[str] = set()

        _LOGGER.debug("Loading sentences...")
        num_sentences = 0
        num_too_short = 0
        num_too_long = 0

        for line in text_iter:
            line = line.strip()
            if not line:
                continue

            for sentence in lang.tokenizer.tokenize(line, guess_pos=guess_pos):
                num_sentences += 1

                # Don't consider non-words in length calculation
                only_words = [
                    w for w in sentence.clean_words if lang.tokenizer.is_word(w)
                ]

                if (min_length is not None) and (len(only_words) < min_length):
                    # Below minimum length
                    num_too_short += 1
                    continue

                if (max_length is not None) and (len(only_words) > max_length):
                    # Above maximum length
                    num_too_long += 1
                    continue

                clean_sentences.append(sentence)

                # Collect missing words
                for word in only_words:
                    if word not in lexicon:
                        missing_words.add(word)

            _LOGGER.debug(line)

        # Report stats
        if num_too_short > 0:
            _LOGGER.warning(
                "Dropped %s/%s sentence(s) for < %s word(s)",
                num_too_short,
                num_sentences,
                min_length,
            )

        if num_too_long > 0:
            _LOGGER.warning(
                "Dropped %s/%s sentence(s) for > %s word(s)",
                num_too_long,
                num_sentences,
                max_length,
            )

        # Guess missing words in a single pass
        if missing_words:
            _LOGGER.debug("Guessing pronunciations for %s word(s)", len(missing_words))

            # Add words to lexicon
            for word, word_phonemes in lang.phonemizer.predict(missing_words, nbest=1):
                lexicon[word] = [
                    remove_stress(WordPronunciation(word_phonemes), keep_stress)
                ]

        # Phonemize missing sentences
        for sentence in clean_sentences:
            # Not currently using breaks
            sentence_prons = lang.phonemizer.phonemize(
                sentence.tokens,
                word_breaks=word_breaks,
                minor_breaks=False,
                major_breaks=False,
                guess_word=lambda word: None,
            )

            if sentence_prons:
                # Choose first pronunciation only
                first_pronunciation = []
                skip_sentence = False
                for wp in sentence_prons:
                    if not wp:
                        skip_sentence = True
                        break

                    first_pronunciation.append(wp[0])

                if skip_sentence:
                    _LOGGER.warning(
                        "Skipping sentence due to missing pronunciations: %s",
                        sentence.raw_text,
                    )
                    continue

                sentences.append(
                    PronouncedSentence(
                        sentence=sentence, pronunciations=[first_pronunciation]
                    )
                )

        _LOGGER.debug("Loaded %s sentence(s)", len(sentences))

        if cache_file_path:
            _LOGGER.debug("Caching sentences in %s", cache_file_path)
            with open(cache_file_path, "w") as cache_file:
                writer = jsonlines.Writer(cache_file)
                for pron_sentence in sentences:
                    writer.write(pron_sentence.to_json(ensure_ascii=False))

    # -------------------------------------------------------------------------

    def score(pair_counter):
        """Compute score for diphone counts"""
        final_score = 0
        for diphone_count in pair_counter.values():
            adjusted_count = diphone_count
            if max_examples > 0:
                adjusted_count = min(max_examples, adjusted_count)

            final_score += adjusted_count

        return final_score

    def coverage(pair_counter):
        """Compute diphone coverage"""
        num_pairs = 0
        for adjusted_count in pair_counter.values():
            if adjusted_count > 0:
                num_pairs += 1

        return num_pairs / len(all_pairs)

    # -------------------------------------------------------------------------

    # Diphone counts for all sentence.
    # Same ordring as sentences list.
    sentence_pair_counters: typing.List[typing.Counter[DIPHONE_TYPE]] = []

    # Index of best sentence found (pre-optimization)
    best_sentence_idx: typing.Optional[int] = None

    # Score of best sentence (pre-optimization)
    best_score: int = 0

    all_sentences_counts: typing.Counter[DIPHONE_TYPE] = Counter()
    sentence_scores: typing.List[float] = []
    sentence_phonemes: typing.Set[str] = set()

    # Get diphone counts for all sentences.
    # Cap pair counts by max-examples (if > 0) to reward breadth.
    #
    # Keep a count multiplier that is inversely proportional to the number of
    # pronunciations the sentence originally had. Sentences with fewer
    # pronunciations count more because there is less uncertainty about which
    # phones will be spoken during recording.
    for sentence_idx, pron_sentence in enumerate(sentences):
        pair_counts: typing.Counter[DIPHONE_TYPE] = Counter()

        for sentence_pron in pron_sentence.pronunciations:
            clean_phonemes = []
            if silence_phone:
                # Beginning of sentence
                clean_phonemes.append(_SILENCE_PHONE)

            # Add example words indexes for pair
            for word_pron in sentence_pron:
                word_pron = remove_stress(word_pron, keep_stress)
                clean_phonemes.extend(word_pron.phonemes)
                sentence_phonemes.update(word_pron.phonemes)

            if silence_phone:
                # End of sentence
                clean_phonemes.append(_SILENCE_PHONE)

            # Add pair counts, associating each pair with a word
            for phoneme_pair in pairwise(clean_phonemes):
                pair_counts[phoneme_pair] += 1

                # Add to counts for all sentences
                all_sentences_counts[phoneme_pair] += 1

                # Merge with lexicon counts to avoid coverage > 1.0
                all_pairs.add(phoneme_pair)

        sentence_pair_counters.append(pair_counts)

        # Find best scoring sentence
        sentence_score = score(pair_counts)
        if sentence_score > best_score:
            best_sentence_idx = sentence_idx
            best_score = sentence_score

        sentence_scores.append(sentence_score)

    # -------------------------------------------------------------------------

    extra_phonemes = sentence_phonemes - phonemes
    if extra_phonemes:
        _LOGGER.warning("Extra phonemes: %s", extra_phonemes)

    _LOGGER.debug(
        "Best possible coverage: single=%s, pair=%s",
        len(sentence_phonemes) / len(phonemes),
        coverage(all_sentences_counts),
    )

    # Begin optimization

    # Start with best scoring sentence
    assert best_sentence_idx is not None, "No best sentence"
    sentences_so_far = set([best_sentence_idx])
    best_sentences = set(sentences_so_far)

    pair_counts_so_far = sentence_pair_counters[best_sentence_idx]
    best_pair_counts: typing.Counter[DIPHONE_TYPE] = Counter(pair_counts_so_far)

    # Initial score
    score_so_far = score(pair_counts_so_far)
    best_score = score_so_far

    # Indexes of all possible snetences
    all_sentence_idxs = set(range(len(sentences)))

    # Do optimization passes until no change or maximum passes
    for pass_idx in range(max_passes):
        score_changed = False
        _LOGGER.debug(
            "Pass %s (pair coverage=%s, N=%s/%s, score=%s)",
            pass_idx + 1,
            coverage(pair_counts_so_far),
            len(sentences_so_far),
            len(sentences),
            score_so_far,
        )

        # Remove extraneous sentences.
        # Force sentences from keep list to remain.
        maybe_remove_idxs = list(sentences_so_far)
        random.shuffle(maybe_remove_idxs)

        _LOGGER.debug(
            "Looking for sentences to remove (%s candidate(s))...",
            len(maybe_remove_idxs),
        )

        num_removed = 0

        for maybe_remove_idx in maybe_remove_idxs:
            removed_pair_counts = sentence_pair_counters[maybe_remove_idx]
            maybe_pair_counts = Counter(pair_counts_so_far)

            # Subtract pair counts assuming the sentence was removed
            for removed_pair, removed_count in removed_pair_counts.items():
                current_count = pair_counts_so_far.get(removed_pair)
                if current_count:
                    maybe_pair_counts[removed_pair] = max(
                        0, current_count - removed_count
                    )

            # Determine if removal would impact score
            maybe_score = score(maybe_pair_counts)
            if maybe_score >= score_so_far:
                # Drop sentence
                sentences_so_far.remove(maybe_remove_idx)
                pair_counts_so_far = maybe_pair_counts
                score_so_far = maybe_score
                score_changed = True
                num_removed += 1

        if num_removed > 0:
            _LOGGER.debug("Removed %s sentence(s)", num_removed)

        # Add new sentences
        maybe_add_idxs = list(all_sentence_idxs - sentences_so_far)
        random.shuffle(maybe_add_idxs)
        _LOGGER.debug(
            "Looking for sentences to add (%s candidate(s))...", len(maybe_add_idxs)
        )

        num_added = 0

        for maybe_add_idx in maybe_add_idxs:
            added_pair_counts = sentence_pair_counters[maybe_add_idx]

            # Merge pair counts assuming sentence was added
            maybe_pair_counts = Counter(pair_counts_so_far)
            for added_pair, added_count in added_pair_counts.items():
                current_count = pair_counts_so_far.get(added_pair)
                if current_count:
                    maybe_pair_counts[added_pair] = current_count + added_count
                else:
                    maybe_pair_counts[added_pair] = added_count

            # Determine if addition would impact score
            maybe_score = score(maybe_pair_counts)
            if maybe_score > score_so_far:
                # Add sentence
                sentences_so_far.add(maybe_add_idx)
                pair_counts_so_far = maybe_pair_counts
                score_so_far = maybe_score
                score_changed = True
                num_added += 1

        if num_added > 0:
            _LOGGER.debug("Added %s sentence(s)", num_added)

        # Keep maximum sentence limit
        if (max_sentences is not None) and (len(sentences_so_far) > max_sentences):
            num_dropped = len(sentences_so_far) - max_sentences

            # Randomly drop sentences.
            # This seems to perform better than other approaches I've tried.
            sentences_to_drop = set(random.sample(sentences_so_far, num_dropped))

            # Re-compute all scores
            sentences_so_far = sentences_so_far - sentences_to_drop
            assert len(sentences_so_far) <= max_sentences

            # Re-calculate pair counts
            pair_counts_so_far = Counter()
            for sentence_idx in sentences_so_far:
                pair_counts_so_far += sentence_pair_counters[sentence_idx]

            new_score = score(pair_counts_so_far)
            if new_score != score_so_far:
                score_changed = True

            _LOGGER.debug(
                "Dropped %s sentence(s) (score %s -> %s)",
                num_dropped,
                score_so_far,
                new_score,
            )

            score_so_far = new_score

        # Check for best score
        if score_so_far > best_score:
            best_score = score_so_far
            best_sentences = set(sentences_so_far)
            best_pair_counts = Counter(pair_counts_so_far)

        # Check if score has changed
        if not score_changed:
            # Optimization complete
            _LOGGER.debug("No change.")
            break

    # Done
    single_phonemes: typing.Set[str] = set()
    for pair in best_pair_counts:
        single_phonemes.update(pair)

    single_coverage = len(single_phonemes) / len(phonemes)
    pair_coverage = coverage(best_pair_counts)

    _LOGGER.debug(
        "Best results (single coverage=%s, pair coverage=%s, N=%s/%s, score=%s)",
        single_coverage,
        pair_coverage,
        len(best_sentences),
        len(sentences),
        best_score,
    )

    return OptimalSentences(
        sentences=[sentences[idx] for idx in best_sentences],
        single_coverage=single_coverage,
        pair_coverage=pair_coverage,
        pair_score=best_score,
        pair_counts=best_pair_counts,
    )


# -------------------------------------------------------------------

REGEX_STRESS = re.compile(f"[{IPA.STRESS_PRIMARY}{IPA.STRESS_SECONDARY}]")


def remove_stress(
    word_pron: WordPronunciation, keep_stress: bool = False
) -> WordPronunciation:
    """Optionally remove stress from a word pronunciation"""
    if keep_stress:
        return word_pron

    word_pron.phonemes = [REGEX_STRESS.sub("", p) for p in word_pron.phonemes]

    return word_pron
