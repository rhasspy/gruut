"""Utility methods for gruut"""
import itertools
import logging
import os
import re
import typing
import xml.etree.ElementTree as etree
from pathlib import Path

import networkx as nx

from gruut.const import (
    DATA_PROP,
    KNOWN_LANGS,
    LANG_ALIASES,
    NODE_TYPE,
    EndElement,
    GraphType,
    Node,
)

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("gruut.utils")

# -----------------------------------------------------------------------------
# Language utilities
# -----------------------------------------------------------------------------

LANG_SPLIT_PATTERN = re.compile(r"[-_]")


def resolve_lang(lang: str) -> str:
    """
    Try to resolve language using aliases.

    Args:
        lang: Language name or alias

    Returns:
        Resolved language name
    """
    lang = lang.lower()
    lang = LANG_ALIASES.get(lang, lang)

    if lang not in KNOWN_LANGS:
        # Try with _ replaced by -
        if "_" in lang:
            lang_parts = lang.split("_")
            maybe_lang = f"{lang_parts[0]}-{lang_parts[1]}"

            if maybe_lang in KNOWN_LANGS:
                lang = maybe_lang

    return lang


def find_lang_dir(
    lang: str,
    search_dirs: typing.Optional[typing.Iterable[typing.Union[str, Path]]] = None,
) -> typing.Optional[Path]:
    """
    Search for a language's model directory by name.

    Tries to find a directory by:

    #. Importing a module name ``gruut_lang_<short_lang>`` where short_lang is "en" for "en-us", etc.
    #. Looking for ``<lang>/lexicon.db`` in each directory in order:

       * ``search_dirs``
       * ``$XDG_CONFIG_HOME/gruut``
       * A "data" directory inside gruut module
       * A "data" directory next to the gruut module

    Args:
        lang: Full language name (e.g., en-us)
        search_dirs: Optional iterable of directory paths to search first

    Returns:
        Path to the language model directory or None if it can't be found
    """
    try:
        base_lang = LANG_SPLIT_PATTERN.split(lang)[0].lower()
        lang_module_name = f"gruut_lang_{base_lang}"
        lang_module = __import__(lang_module_name)

        _LOGGER.debug("(%s) successfully imported %s", lang, lang_module_name)

        return lang_module.get_lang_dir()
    except ImportError:
        _LOGGER.debug("(%s) unable to import module", lang)
        pass

    search_dirs = typing.cast(typing.List[Path], [Path(p) for p in search_dirs or []])

    # ${XDG_CONFIG_HOME}/gruut or ${HOME}/gruut
    maybe_config_home = os.environ.get("XDG_CONFIG_HOME")
    if maybe_config_home:
        search_dirs.append(Path(maybe_config_home) / "gruut")
    else:
        search_dirs.append(Path.home() / ".config" / "gruut")

    # Data directory *inside* gruut
    search_dirs.append(_DIR / "data")

    # Data directory *next to* gruut
    search_dirs.append(_DIR.parent / "data")

    _LOGGER.debug("(%s) searching %s for language file(s)", lang, search_dirs)

    for check_dir in search_dirs:
        lang_dir = check_dir / lang
        lexicon_path = lang_dir / "lexicon.db"
        if lexicon_path.is_file():
            _LOGGER.debug("(%s) found language file(s) in %s", lang, lang_dir)
            return lang_dir

    return None


# -----------------------------------------------------------------------------
# Babel
# -----------------------------------------------------------------------------


def get_currency_names(locale_str: str) -> typing.Dict[str, str]:
    """
    Try to get currency names and symbols for a Babel locale.

    Returns:
        Dictionary whose keys are currency symbols (like "$") and whose values are currency names (like "USD")
    """
    currency_names = {}

    try:
        import babel
        import babel.numbers

        locale = babel.Locale(locale_str)
        currency_names = {
            babel.numbers.get_currency_symbol(cn): cn for cn in locale.currency_symbols
        }
    except ImportError:
        # Expected if babel is not installed
        pass
    except Exception:
        _LOGGER.warning("get_currency_names")

    return currency_names


# -----------------------------------------------------------------------------
# Iteration
# -----------------------------------------------------------------------------


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def sliding_window(iterable, n=2):
    """Returns a sliding window of size n over an iterable"""
    iterables = itertools.tee(iterable, n)

    for win_iter, num_skipped in zip(iterables, itertools.count()):
        for _ in range(num_skipped):
            next(win_iter, None)

    return zip(*iterables)


# -----------------------------------------------------------------------------
# XML
# -----------------------------------------------------------------------------

NO_NAMESPACE_PATTERN = re.compile(r"^{[^}]+}")


def tag_no_namespace(tag: str) -> str:
    """Remove namespace from XML tag"""
    return NO_NAMESPACE_PATTERN.sub("", tag)


def attrib_no_namespace(
    element: etree.Element, name: str, default: typing.Any = None
) -> typing.Any:
    """Search for an attribute by key without namespaces"""
    for key, value in element.attrib.items():
        key_no_ns = NO_NAMESPACE_PATTERN.sub("", key)
        if key_no_ns == name:
            return value

    return default


def text_and_elements(element, is_last=False):
    """Yields element, text, sub-elements, end element, and tail"""
    element_metadata = None

    if is_last:
        # True if this is the last child element of a parent.
        # Used to preserve whitespace.
        element_metadata = {"is_last": True}

    yield element, element_metadata

    # Text before any tags (or end tag)
    text = element.text if element.text is not None else ""
    if text.strip():
        yield text

    children = list(element)
    last_child_idx = len(children) - 1

    for child_idx, child in enumerate(children):
        # Sub-elements
        is_last = child_idx == last_child_idx
        yield from text_and_elements(child, is_last=is_last)

    # End of current element
    yield EndElement(element)

    # Text after the current tag
    tail = element.tail if element.tail is not None else ""
    if tail.strip():
        yield tail


# -----------------------------------------------------------------------------
# Text
# -----------------------------------------------------------------------------

NON_WORDS_PATTERN = re.compile(r"\W")


def remove_non_word_chars(s: str) -> str:
    """Removes non-word characters from a string"""
    return NON_WORDS_PATTERN.sub("", s)


# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------


def print_graph(
    graph: GraphType,
    node: typing.Union[NODE_TYPE, Node],
    indent: str = "--",
    level: int = 1,
    print_func=print,
):
    """Prints a graph to the console"""
    if isinstance(node, Node):
        n_data = node
        graph_node = node.node
    else:
        graph_node = node
        n_data = typing.cast(Node, graph.nodes[graph_node][DATA_PROP])

    print_func(indent * level, graph_node, n_data)
    for succ_node in graph.successors(graph_node):
        print_graph(
            graph, succ_node, indent=indent, level=level + 1, print_func=print_func
        )


def leaves(graph: GraphType, node: Node):
    """Iterate through the leaves of a graph in depth-first order"""
    for dfs_node in nx.dfs_preorder_nodes(graph, node.node):
        if not graph.out_degree(dfs_node) == 0:
            continue

        yield graph.nodes[dfs_node][DATA_PROP]


def pipeline_split(
    split_func, graph: GraphType, parent_node: Node,
):
    """Splits leaf nodes of tree into zero or more sub-nodes"""
    for leaf_node in list(leaves(graph, parent_node)):
        for node_class, node_kwargs in split_func(graph, leaf_node):
            new_node = node_class(node=len(graph), **node_kwargs)
            graph.add_node(new_node.node, data=new_node)
            graph.add_edge(leaf_node.node, new_node.node)


def pipeline_transform(
    transform_func, graph: GraphType, parent_node: Node,
):
    """Transforms leaves of tree with a custom function"""
    for leaf_node in list(leaves(graph, parent_node)):
        transform_func(graph, leaf_node)
