#!/usr/bin/env python3
"""Guess word pronunciations using a Phonetisaurus FST

See bin/fst2npz.py to convert an FST to a numpy graph.
"""
import argparse
import logging
import os
import sys
import time
import typing
from collections import defaultdict
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger("g2p_phonetisaurus")

NUMPY_GRAPH = typing.Dict[str, np.ndarray]

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="g2p_phonetisaurus")

    # Create subparsers for each sub-command
    sub_parsers = parser.add_subparsers()
    sub_parsers.required = True
    sub_parsers.dest = "command"

    # -------
    # Predict
    # -------
    predict_parser = sub_parsers.add_parser(
        "predict", help="Predict phonemes for word(s)"
    )
    predict_parser.add_argument(
        "--graph", required=True, help="Path to graph npz file from fst2npy.py"
    )
    predict_parser.add_argument(
        "words", nargs="*", help="Words to guess pronunciations for"
    )
    predict_parser.add_argument(
        "--max-guesses",
        default=1,
        type=int,
        help="Maximum number of guesses per word (default: 1)",
    )
    predict_parser.add_argument(
        "--beam",
        default=500,
        type=int,
        help="Initial width of search beam (default: 500)",
    )
    predict_parser.add_argument(
        "--min-beam",
        default=100,
        type=int,
        help="Minimum width of search beam (default: 100)",
    )
    predict_parser.add_argument(
        "--beam-scale",
        default=0.6,
        type=float,
        help="Scalar multiplied by beam after each step (default: 0.6)",
    )
    predict_parser.add_argument(
        "--grapheme-separator",
        default="",
        help="Separator between input graphemes (default: none)",
    )
    predict_parser.add_argument(
        "--phoneme-separator",
        default=" ",
        help="Separator between output phonemes (default: space)",
    )
    predict_parser.add_argument(
        "--preload-graph",
        action="store_true",
        help="Preload graph into memory before starting",
    )
    predict_parser.set_defaults(func=do_predict)

    # ----
    # Test
    # ----
    test_parser = sub_parsers.add_parser("test", help="Test G2P model on a lexicon")
    test_parser.add_argument(
        "--graph", required=True, help="Path to graph npz file from fst2npy.py"
    )
    test_parser.add_argument(
        "texts", nargs="*", help="Lines with '<word> <phoneme> <phoneme> ...'"
    )
    test_parser.add_argument(
        "--beam",
        default=500,
        type=int,
        help="Initial width of search beam (default: 500)",
    )
    test_parser.add_argument(
        "--min-beam",
        default=100,
        type=int,
        help="Minimum width of search beam (default: 100)",
    )
    test_parser.add_argument(
        "--beam-scale",
        default=0.6,
        type=float,
        help="Scalar multiplied by beam after each step (default: 0.6)",
    )
    test_parser.add_argument(
        "--preload-graph",
        action="store_true",
        help="Preload graph into memory before starting",
    )
    test_parser.set_defaults(func=do_test)

    # ----------------
    # Shared arguments
    # ----------------
    for sub_parser in [predict_parser, test_parser]:
        sub_parser.add_argument(
            "--debug", action="store_true", help="Print DEBUG messages to console"
        )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    args.func(args)


# -----------------------------------------------------------------------------


def do_predict(args):
    """Predict phonemes for words"""
    args.graph = Path(args.graph)

    _LOGGER.debug("Loading graph from %s", args.graph)
    phon_graph = PhonetisaurusGraph.load(args.graph, preload=args.preload_graph)

    if args.words:
        # Arguments
        words = args.words
        _LOGGER.info("Guessing pronunciations for %s word(s)", len(words))
    else:
        # Standard input
        words = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading words from stdin...", file=sys.stderr)

    # Guess pronunciations
    for word, graphemes, phonemes in phon_graph.g2p(
        words,
        grapheme_separator=args.grapheme_separator,
        max_guesses=args.max_guesses,
        beam=args.beam,
        min_beam=args.min_beam,
        beam_scale=args.beam_scale,
    ):
        if not phonemes:
            _LOGGER.warning("No pronunciation for %s (%s)", word, graphemes)
            continue

        print(word, args.phoneme_separator.join(phonemes))


# -----------------------------------------------------------------------------


def do_test(args):
    """Test performance relative a known lexicon"""
    try:
        from rapidfuzz.string_metric import levenshtein
    except ImportError as e:
        _LOGGER.critical("rapidfuzz library is needed for levenshtein distance")
        _LOGGER.critical("pip install 'rapidfuzz>=1.4.1'")
        raise e

    args.graph = Path(args.graph)

    _LOGGER.debug("Loading graph from %s", args.graph)
    phon_graph = PhonetisaurusGraph.load(args.graph, preload=args.preload_graph)

    if args.texts:
        lines = args.texts
    else:
        lines = sys.stdin

        if os.isatty(sys.stdin.fileno()):
            print("Reading lexicon lines from stdin...", file=sys.stderr)

    # Load lexicon
    lexicon = {}
    for line in lines:
        line = line.strip()
        if (not line) or (" " not in line):
            continue

        word, actual_phonemes = line.split(maxsplit=1)
        lexicon[word] = actual_phonemes

    # Predict phonemes
    predicted_phonemes = {}
    start_time = time.perf_counter()

    for word in lexicon:
        for _, _, guessed_phonemes in phon_graph.g2p(
            [word],
            beam=args.beam,
            min_beam=args.min_beam,
            beam_scale=args.beam_scale,
            max_guesses=1,
        ):
            predicted_phonemes[word] = " ".join(guessed_phonemes)

            # Only one guess
            break

    end_time = time.perf_counter()

    # Calculate PER
    num_errors = 0
    num_missing = 0
    num_phonemes = 0

    for word, actual_phonemes in lexicon.items():
        expected_phonemes = predicted_phonemes.get(word, "")

        if expected_phonemes:
            distance = levenshtein(expected_phonemes, actual_phonemes)
            num_errors += distance
            num_phonemes += len(actual_phonemes)
        else:
            num_missing += 1
            _LOGGER.warning("No pronunciation for %s", word)

    assert num_phonemes > 0, "No phonemes were read"

    # Calculate results
    per = round(num_errors / num_phonemes, 2)
    wps = round(len(predicted_phonemes) / (end_time - start_time), 2)
    print("PER:", per, "Errors:", num_errors, "words/sec:", wps)

    if num_missing > 0:
        print("Total missing:", num_missing)


# -----------------------------------------------------------------------------

_NOT_FINAL = object()


class PhonetisaurusGraph:
    """Graph of numpy arrays that represents a Phonetisaurus FST

    Also contains shared cache of edges and final state probabilities.
    These caches are necessary to ensure that the .npz file stays small and fast
    to load.
    """

    def __init__(self, graph: NUMPY_GRAPH, preload: bool = False):
        self.graph = graph

        self.start_node = int(self.graph["start_node"].item())

        # edge_index -> (from_node, to_node, ilabel, olabel)
        self.edges = self.graph["edges"]
        self.edge_probs = self.graph["edge_probs"]

        # int -> [str]
        self.symbols = []
        for symbol_str in self.graph["symbols"]:
            symbol_list = symbol_str.replace("_", "").split("|")
            self.symbols.append((len(symbol_list), symbol_list))

        # nodes that are accepting states
        self.final_nodes = self.graph["final_nodes"]

        # node -> probability
        self.final_probs = self.graph["final_probs"]

        # Cache
        self.preloaded = preload
        self.out_edges: typing.Dict[int, typing.List[int]] = defaultdict(list)
        self.final_node_probs: typing.Dict[int, typing.Any] = {}

        if preload:
            # Load out edges
            for edge_idx, (from_node, *_) in enumerate(self.edges):
                self.out_edges[from_node].append(edge_idx)

            # Load final probabilities
            self.final_node_probs.update(zip(self.final_nodes, self.final_probs))

    @staticmethod
    def load(graph_path: typing.Union[str, Path], **kwargs) -> "PhonetisaurusGraph":
        """Load .npz file with numpy graph"""
        np_graph = np.load(graph_path, allow_pickle=True)
        return PhonetisaurusGraph(np_graph, **kwargs)

    def g2p(
        self, words: typing.Iterable[typing.Union[str, typing.Sequence[str]]], **kwargs
    ) -> typing.Iterable[
        typing.Tuple[
            typing.Union[str, typing.Sequence[str]],
            typing.Sequence[str],
            typing.Sequence[str],
        ],
    ]:
        """Guess phonemes for words"""
        for word in words:
            for graphemes, phonemes in self.g2p_one(word, **kwargs):
                yield word, graphemes, phonemes

    def g2p_one(
        self,
        word: typing.Union[str, typing.Sequence[str]],
        eps: str = "<eps>",
        beam: int = 5000,
        min_beam: int = 100,
        beam_scale: float = 0.6,
        grapheme_separator: str = "",
        max_guesses: int = 1,
    ) -> typing.Iterable[typing.Tuple[typing.Sequence[str], typing.Sequence[str]]]:
        """Guess phonemes for word"""
        current_beam = beam
        graphemes: typing.Sequence[str] = []

        if isinstance(word, str):
            word = word.strip()

            if grapheme_separator:
                graphemes = word.split(grapheme_separator)
            else:
                graphemes = list(word)
        else:
            graphemes = word

        if not graphemes:
            return graphemes, []

        # (prob, node, graphemes, phonemes, final, beam)
        q: typing.List[
            typing.Tuple[
                float,
                typing.Optional[int],
                typing.Sequence[str],
                typing.List[str],
                bool,
            ]
        ] = [(0.0, self.start_node, graphemes, [], False)]

        q_next: typing.List[
            typing.Tuple[
                float,
                typing.Optional[int],
                typing.Sequence[str],
                typing.List[str],
                bool,
            ]
        ] = []

        # (prob, phonemes)
        best_heap: typing.List[typing.Tuple[float, typing.Sequence[str]]] = []

        # Avoid duplicate guesses
        guessed_phonemes: typing.Set[typing.Tuple[str, ...]] = set()

        while q:
            done_with_word = False
            q_next = []

            for prob, node, next_graphemes, output, is_final in q:
                if is_final:
                    # Complete guess
                    phonemes = tuple(output)
                    if phonemes not in guessed_phonemes:
                        best_heap.append((prob, phonemes))
                        guessed_phonemes.add(phonemes)

                    if len(best_heap) >= max_guesses:
                        done_with_word = True
                        break

                    continue

                assert node is not None

                if not next_graphemes:
                    if self.preloaded:
                        final_prob = self.final_node_probs.get(node, _NOT_FINAL)
                    else:
                        final_prob = self.final_node_probs.get(node)
                        if final_prob is None:
                            final_idx = int(np.searchsorted(self.final_nodes, node))
                            if self.final_nodes[final_idx] == node:
                                # Cache
                                final_prob = float(self.final_probs[final_idx])
                                self.final_node_probs[node] = final_prob
                            else:
                                # Not a final state
                                final_prob = _NOT_FINAL
                                self.final_node_probs[node] = final_prob

                    if final_prob != _NOT_FINAL:
                        final_prob = typing.cast(float, final_prob)
                        q_next.append((prob + final_prob, None, [], output, True))

                len_next_graphemes = len(next_graphemes)
                if self.preloaded:
                    # Was pre-loaded in __init__
                    edge_idxs = self.out_edges[node]
                else:
                    # Build cache during search
                    maybe_edge_idxs = self.out_edges.get(node)
                    if maybe_edge_idxs is None:
                        edge_idx = int(np.searchsorted(self.edges[:, 0], node))
                        edge_idxs = []
                        while self.edges[edge_idx][0] == node:
                            edge_idxs.append(edge_idx)
                            edge_idx += 1

                        # Cache
                        self.out_edges[node] = edge_idxs
                    else:
                        edge_idxs = maybe_edge_idxs

                for edge_idx in edge_idxs:
                    _, to_node, ilabel_idx, olabel_idx = self.edges[edge_idx]
                    out_prob = self.edge_probs[edge_idx]

                    len_igraphemes, igraphemes = self.symbols[ilabel_idx]

                    if len_igraphemes > len_next_graphemes:
                        continue

                    if igraphemes == [eps]:
                        item = (prob + out_prob, to_node, next_graphemes, output, False)
                        q_next.append(item)
                    else:
                        sub_graphemes = next_graphemes[:len_igraphemes]
                        if igraphemes == sub_graphemes:
                            _, olabel = self.symbols[olabel_idx]
                            item = (
                                prob + out_prob,
                                to_node,
                                next_graphemes[len(sub_graphemes) :],
                                output + olabel,
                                False,
                            )
                            q_next.append(item)

            if done_with_word:
                break

            q_next = sorted(q_next, key=lambda item: item[0])[:current_beam]
            q = q_next

            current_beam = max(min_beam, (int(current_beam * beam_scale)))

        # Yield guesses
        if best_heap:
            for _, guess_phonemes in sorted(best_heap, key=lambda item: item[0])[
                :max_guesses
            ]:
                yield graphemes, [p for p in guess_phonemes if p]
        else:
            # No guesses
            yield graphemes, []


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
