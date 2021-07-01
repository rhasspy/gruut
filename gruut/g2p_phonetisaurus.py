#!/usr/bin/env python3
"""Guess word pronunciations using a Phonetisaurus FST

See bin/fst2npz.py to convert an FST to a numpy graph.
"""
import argparse
import heapq
import logging
import os
import sys
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
    parser.add_argument("graph", help="Path to graph npz file from fst2npy.py")
    parser.add_argument("words", nargs="*", help="Words to guess pronunciations for")
    parser.add_argument(
        "--max-guesses",
        default=1,
        type=int,
        help="Maximum number of guesses per word (default: 1)",
    )
    parser.add_argument(
        "--beam",
        default=5000,
        type=int,
        help="Initial width of search beam (default: 5000)",
    )
    parser.add_argument(
        "--min-beam",
        default=100,
        type=int,
        help="Minimum width of search beam (default: 100)",
    )
    parser.add_argument(
        "--beam-scale",
        default=0.6,
        type=float,
        help="Scalar multiplied by beam after each step (default: 0.6)",
    )
    parser.add_argument(
        "--grapheme-separator",
        default="",
        help="Separator between input graphemes (default: none)",
    )
    parser.add_argument(
        "--phoneme-separator",
        default=" ",
        help="Separator between output phonemes (default: space)",
    )
    args = parser.parse_args()
    args.graph = Path(args.graph)
    args.words = args.words

    logging.basicConfig(level=logging.INFO)

    _LOGGER.info("Loading graph from %s", args.graph)
    phon_graph = PhonetisaurusGraph.load(args.graph)

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
        beam_scale=args.beam_scale,
    ):
        if not phonemes:
            _LOGGER.warning("No pronunciation for %s (%s)", word, graphemes)
            continue

        print(word, args.phoneme_separator.join(phonemes))


# -----------------------------------------------------------------------------

_NOT_FINAL = object()


class PhonetisaurusGraph:
    """Graph of numpy arrays that represents a Phonetisaurus FST

    Also contains shared cache of edges and final state probabilities.
    These caches are necessary to ensure that the .npz file stays small and fast
    to load.
    """

    def __init__(self, graph: NUMPY_GRAPH):
        self.graph = graph

        self.start_node = int(self.graph["start_node"].item())

        # edge_index -> (from_node, to_node, ilabel, olabel)
        self.edges = self.graph["edges"]
        self.edge_probs = self.graph["edge_probs"]

        # int -> str
        self.symbols = self.graph["symbols"]

        # nodes that are accepting states
        self.final_nodes = self.graph["final_nodes"]

        # node -> probability
        self.final_probs = self.graph["final_probs"]

        # Cache
        self.out_edges: typing.Dict[int, typing.List[int]] = defaultdict(list)
        self.final_node_probs: typing.Dict[int, float] = {}

    @staticmethod
    def load(graph_path: typing.Union[str, Path]) -> "PhonetisaurusGraph":
        """Load .npz file with numpy graph"""
        np_graph = np.load(graph_path, allow_pickle=True)
        return PhonetisaurusGraph(np_graph)

    def g2p(
        self,
        words: typing.Iterable[typing.Union[str, typing.Sequence[str]]],
        eps: str = "<eps>",
        beam: int = 5000,
        min_beam: int = 100,
        beam_scale: float = 0.6,
        grapheme_separator: str = "",
        max_guesses: int = 1,
    ):
        """Guess phonemes for words"""
        for word in words:
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
                continue

            # (prob, node, graphemes, phonemes, final, beam)
            q: typing.List[
                typing.Tuple[float, int, typing.Sequence[str], typing.List[str], bool]
            ] = [(0.0, self.start_node, graphemes, [], False)]

            q_next: typing.List[
                typing.Tuple[float, int, typing.Sequence[str], typing.List[str], bool]
            ] = []

            # (prob, phonemes)
            best_heap: typing.List[typing.Tuple[float, typing.List[str]]] = []

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
                            heapq.heappush(  # type: ignore
                                best_heap, (prob, phonemes)
                            )
                            guessed_phonemes.add(phonemes)

                        if len(best_heap) >= max_guesses:
                            done_with_word = True
                            break

                        continue

                    if not next_graphemes:
                        final_prob: typing.Any = self.final_node_probs.get(node)
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
                            assert isinstance(final_prob, float)
                            heapq.heappush(  # type: ignore
                                q_next, (prob + final_prob, None, None, output, True)
                            )

                    edge_idxs = self.out_edges.get(node)
                    if edge_idxs is None:
                        edge_idx = int(np.searchsorted(self.edges[:, 0], node))
                        edge_idxs = []
                        while self.edges[edge_idx][0] == node:
                            edge_idxs.append(edge_idx)
                            edge_idx += 1

                        # Cache
                        self.out_edges[node] = edge_idxs

                    for edge_idx in edge_idxs:
                        _, to_node, ilabel_idx, olabel_idx = self.edges[edge_idx]
                        out_prob = self.edge_probs[edge_idx]

                        igraphemes = self.symbols[ilabel_idx].split("|")

                        if len(igraphemes) > len(next_graphemes):
                            continue

                        if igraphemes == [eps]:
                            item = (
                                prob + out_prob,
                                to_node,
                                next_graphemes,
                                output,
                                False,
                            )
                            heapq.heappush(q_next, item)
                        else:
                            sub_graphemes = next_graphemes[: len(igraphemes)]
                            if igraphemes == sub_graphemes:
                                olabel = (
                                    self.symbols[olabel_idx].replace("_", "").split("|")
                                )
                                item = (
                                    prob + out_prob,
                                    to_node,
                                    next_graphemes[len(sub_graphemes) :],
                                    output + olabel,
                                    False,
                                )
                                heapq.heappush(q_next, item)

                if done_with_word:
                    break

                q_next = q_next[:current_beam]
                q = q_next

                current_beam = max(min_beam, (int(current_beam * beam_scale)))

            # Yield guesses
            if best_heap:
                for _, guess_phonemes in best_heap[:max_guesses]:
                    yield word, graphemes, [p for p in guess_phonemes if p]
            else:
                # No guesses
                yield word, graphemes, []


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
