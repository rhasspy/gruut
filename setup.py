"""Setup file for gruut"""
from collections import defaultdict
from pathlib import Path

import setuptools

this_dir = Path(__file__).parent
module_dir = this_dir / "gruut"

# -----------------------------------------------------------------------------

# Load README in as long description
long_description: str = ""
readme_path = this_dir / "README.md"
if readme_path.is_file():
    long_description = readme_path.read_text(encoding="UTF-8")

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r", encoding="utf-8") as requirements_file:
        requirements = requirements_file.read().splitlines()

version_path = module_dir / "VERSION"
with open(version_path, "r", encoding="utf-8") as version_file:
    version = version_file.read().strip()

# x.y.z -> x.y.0
base_version = ".".join(version.split(".")[:-1] + ["0"])

# -----------------------------------------------------------------------------
# extras_require
# -----------------------------------------------------------------------------

# dependency => [tags]
extras = {
    "hazm~=0.7.0": ["fa"],
    "conllu>=4.4": ["train"],
    "rapidfuzz>=1.4.1": ["train"],
    "aeneas~=1.7.3.0": ["align"],  # requires numpy to install
    "pydub~=0.24.1": ["align"],
    "mishkal~=0.4.0": ["ar"],
    "codernitydb3~=0.6.0": ["ar"],
    "phonetisaurus~=0.3.0": ["g2p"],
}

# Create language-specific extras
for lang in ["ar", "cs", "de", "es", "fa", "fr", "it", "nl", "pt", "ru", "sv", "sw"]:
    extras[f"gruut_lang_{lang}~={base_version}"] = [lang]

# Add "all" tag
for tags in extras.values():
    tags.append("all")

# Invert for setup
extras_require = defaultdict(list)
for dep, tags in extras.items():
    for tag in tags:
        extras_require[tag].append(dep)


# -----------------------------------------------------------------------------

include_files = {"lexicon.db", "model.crf"}

data_dir = module_dir / "data"
data_files = [
    str(f.relative_to(module_dir))
    for f in data_dir.rglob("*")
    if f.is_file() and (f.name in include_files)
]

setuptools.setup(
    name="gruut",
    description="A tokenizer, text cleaner, and phonemizer for many human languages.",
    version=version,
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    url="https://github.com/rhasspy/gruut",
    packages=setuptools.find_packages(),
    package_data={"gruut": data_files + ["VERSION", "py.typed"]},
    install_requires=requirements,
    extras_require={
        ':python_version<"3.7"': ["dataclasses", "types-dataclasses"],
        **extras_require,
    },
    entry_points={"console_scripts": ["gruut = gruut.__main__:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
