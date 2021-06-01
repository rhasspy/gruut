"""Utility methods for gruut"""
import itertools
import logging
import os
import re
import typing
from pathlib import Path

_DIR = Path(__file__).parent
_LOGGER = logging.getLogger("gruut.utils")


# -----------------------------------------------------------------------------


def find_lang_dir(
    lang: str,
    search_dirs: typing.Optional[typing.Iterable[typing.Union[str, Path]]] = None,
) -> typing.Optional[Path]:
    """Search for a language's model directory by name"""
    try:
        base_lang = lang.split("-")[0]
        lang_module_name = f"gruut_lang_{base_lang}"
        lang_module = __import__(lang_module_name)

        _LOGGER.debug("Successfully imported %s", lang_module_name)

        return lang_module.get_lang_dir()
    except ImportError:
        _LOGGER.debug("Failed to import module for %s. Searching manually.", lang)
        pass

    search_dirs = typing.cast(typing.List[Path], [Path(p) for p in search_dirs or []])

    # ${XDG_CONFIG_HOME}/gruut or ${HOME}/gruut
    maybe_config_home = os.environ.get("XDG_CONFIG_HOME")
    if maybe_config_home:
        search_dirs.append(Path(maybe_config_home) / "gruut")
    else:
        search_dirs.append(Path.home() / ".config" / "gruut")

    # Data directory *next to* gruut
    search_dirs.append(_DIR.parent / "data")

    # Data directory *inside* gruut
    search_dirs.append(_DIR / "data")

    for check_dir in search_dirs:
        lang_dir = check_dir / lang
        lexicon_path = lang_dir / "lexicon.db"
        if lexicon_path.is_file():
            return lang_dir

    return None


# -----------------------------------------------------------------------------


def maybe_compile_regex(str_or_pattern: typing.Union[str, re.Pattern]) -> re.Pattern:
    """Compile regex pattern if it's a string"""
    if isinstance(str_or_pattern, re.Pattern):
        return str_or_pattern

    return re.compile(str_or_pattern)


# -----------------------------------------------------------------------------


def get_currency_names(locale_str: str) -> typing.Dict[str, str]:
    """Try to get currency names and symbols for a Babel locale"""
    currency_names = {}

    try:
        import babel
        import babel.numbers

        locale = babel.Locale(locale_str)
        currency_names = {
            babel.numbers.get_currency_symbol(cn): cn for cn in locale.currency_symbols
        }
    except ImportError:
        pass

    return currency_names


# -----------------------------------------------------------------------------


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)
