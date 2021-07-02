"""Setup file for gruut_lang_nl"""
import os
from pathlib import Path

import setuptools

module_name = "gruut_lang_nl"

this_dir = Path(__file__).parent
module_dir = this_dir / module_name

# -----------------------------------------------------------------------------

version_path = module_dir / "VERSION"
with open(version_path, "r") as version_file:
    version = version_file.read().strip()

# -----------------------------------------------------------------------------

setuptools.setup(
    name=module_name,
    description="Dutch language files for gruut tokenizer/phonemizer",
    version=version,
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    url="https://github.com/rhasspy/gruut",
    packages=setuptools.find_packages(),
    package_data={
        module_name: [
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
