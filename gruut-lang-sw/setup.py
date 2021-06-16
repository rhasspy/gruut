"""Setup file for gruut_lang_sw"""
import os
from pathlib import Path

import setuptools

this_dir = Path(__file__).parent
module_dir = this_dir / "gruut_lang_sw"

# -----------------------------------------------------------------------------

version_path = module_dir / "VERSION"
with open(version_path, "r") as version_file:
    version = version_file.read().strip()

# -----------------------------------------------------------------------------

setuptools.setup(
    name="gruut_lang_sw",
    description="Swahili language files for gruut tokenizer/phonemizer",
    version=version,
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    url="https://github.com/rhasspy/gruut",
    packages=setuptools.find_packages(),
    package_data={
        "gruut_lang_sw": [
            "VERSION",
            "lexicon.db",
            "g2p/model.crf",
            "espeak/lexicon.db",
            "espeak/g2p/model.crf",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
