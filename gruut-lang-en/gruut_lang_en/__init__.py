"""English language resources"""
import os
import typing
from pathlib import Path

try:
    import importlib.resources

    files = importlib.resources.files
except (ImportError, AttributeError):
    # Backport for Python < 3.9
    import importlib_resources  # type: ignore

    files = importlib_resources.files

_PACKAGE = "gruut_lang_en"
_DIR = Path(typing.cast(os.PathLike, files(_PACKAGE)))


def get_lang_dir() -> Path:
    """Get directory with language resources"""
    return _DIR
