"""Russian language resources"""
from pathlib import Path

_DIR = Path(__file__).parent


def get_lang_dir() -> Path:
    """Get directory with language resources"""
    return _DIR
