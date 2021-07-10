"""Setup file for gruut_lang_fr"""
import os
from pathlib import Path

import setuptools

module_name = "gruut_lang_fr"

this_dir = Path(__file__).parent
module_dir = this_dir / module_name

# -----------------------------------------------------------------------------

version_path = module_dir / "VERSION"
with open(version_path, "r") as version_file:
    version = version_file.read().strip()


extra_files = []

pos_model = module_dir / "pos" / "model.crf"
if pos_model.is_file():
    extra_files.append(str(pos_model.relative_to(module_dir)))

# -----------------------------------------------------------------------------

setuptools.setup(
    name=module_name,
    description="French language files for gruut tokenizer/phonemizer",
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
        + extra_files
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
