"""Setup file for gruut"""
import os
from pathlib import Path

import setuptools

this_dir = Path(__file__).parent

# -----------------------------------------------------------------------------

# Load README in as long description
long_description: str = ""
readme_path = this_dir / "README.md"
if readme_path.is_file():
    long_description = readme_path.read_text()

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r") as requirements_file:
        requirements = requirements_file.read().splitlines()

version_path = this_dir / "VERSION"
with open(version_path, "r") as version_file:
    version = version_file.read().strip()

# -----------------------------------------------------------------------------

module_dir = this_dir / "gruut"

setuptools.setup(
    name="gruut",
    description="A tokenizer, text cleaner, and phonemizer for many human languages.",
    version=version,
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    url="https://github.com/rhasspy/gruut",
    packages=setuptools.find_packages(),
    package_data={"gruut": ["py.typed"]},
    install_requires=requirements,
    entry_points={"console_scripts": ["gruut = gruut.__main__:main"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
)
