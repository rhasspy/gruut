"""Setup file for gruut_lang_it"""
from pathlib import Path

import setuptools

module_name = "gruut_lang_it"

this_dir = Path(__file__).parent
module_dir = this_dir / module_name

# -----------------------------------------------------------------------------

# Load README in as long description
long_description: str = ""
readme_path = this_dir / "README.md"
if readme_path.is_file():
    long_description = readme_path.read_text(encoding="utf-8")

version_path = module_dir / "VERSION"
with open(version_path, "r", encoding="utf-8") as version_file:
    version = version_file.read().strip()


# Extra package data files
extra_files = []
maybe_extra_files = ["pos/model.crf", "pos/postagger.model"]
for maybe_extra_str in maybe_extra_files:
    extra_path = module_dir / maybe_extra_str
    if extra_path.is_file():
        extra_files.append(maybe_extra_str)

# -----------------------------------------------------------------------------

setuptools.setup(
    name=module_name,
    description="Italian language files for gruut tokenizer/phonemizer",
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
    long_description=long_description,
    long_description_content_type="text/markdown",
)
