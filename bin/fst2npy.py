#!/usr/bin/env python3
"""Convert a Phonetisaurus FST (printed with fstprint) to numpy arrays"""
import argparse
import logging
import typing
from pathlib import Path

import numpy as np

_LOGGER = logging.getLogger("fst2npy")

# -----------------------------------------------------------------------------


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(prog="fst2npy.py")
    parser.add_argument(
        "fst_text", help="Path to Phonetisaurus text FST (use fstprint)"
    )
    parser.add_argument("npz", help="Path to write numpy npz file")
    args = parser.parse_args()
    args.fst_text = Path(args.fst_text)
    args.npz = Path(args.npz)

    # Create output directory
    args.npz.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    _LOGGER.info("Converting %s to graph", args.fst_text)
    graph = fst2graph(args.fst_text)
    _LOGGER.info("Writing graph to %s", args.npz)
    with open(args.npz, "wb") as npz_file:
        np.savez(npz_file, **graph)


def fst2graph(fst_path: typing.Union[str, Path]) -> typing.Dict[str, np.ndarray]:
    """Read text FST and convert to a graph of numpy arrays"""
    # (from_node, to_node, ilabel, olabel)
    edges = []

    # edge probabilities (same order as edges)
    edge_probs = []

    # nodes that are accepting states
    final_nodes = []

    # final state probabilities (same order as final_nodes)
    final_probs = []

    # str -> int
    symbols: typing.Dict[str, int] = {}

    to_nodes = set()

    with open(fst_path, "r", encoding="utf-8") as fst_file:
        for line in fst_file:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 3:
                # Final state
                node = int(parts[0])
                prob = 0.0 if len(parts) < 2 else float(parts[1])
                final_nodes.append(node)
                final_probs.append(prob)
            else:
                # Transition
                from_node, to_node = int(parts[0]), int(parts[1])
                ilabel, olabel = parts[2], parts[3]
                prob = 0.0 if len(parts) < 5 else float(parts[4])

                ilabel_idx = symbols.get(ilabel, len(symbols))
                symbols[ilabel] = ilabel_idx

                olabel_idx = symbols.get(olabel, len(symbols))
                symbols[olabel] = olabel_idx

                edges.append((from_node, to_node, ilabel_idx, olabel_idx))
                edge_probs.append(prob)

                to_nodes.add(to_node)

    # Determine start node
    start_node: typing.Optional[int] = None
    for from_node, *_ in edges:
        if from_node not in to_nodes:
            start_node = from_node
            break

    assert start_node is not None, "No start node"

    edges = sorted(edges, key=lambda e: e[0])

    return {
        "start_node": np.array([start_node], dtype=np.int32),
        "edges": np.array(edges, dtype=np.int32),
        "edge_probs": np.array(edge_probs, dtype=np.float32),
        "final_nodes": np.array(final_nodes, dtype=np.int32),
        "final_probs": np.array(final_probs, dtype=np.float32),
        "symbols": np.array(
            [k for k, v in sorted(symbols.items(), key=lambda kv: kv[1])], dtype=object
        ),
    }


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
